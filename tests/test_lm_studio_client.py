"""Tests for the LM Studio client in pipeline/translator.py.

Covers the bug this module was rewritten for: LM_STUDIO_URL already ends in
"/v1", so concatenating "/chat/completions" onto it (or using it verbatim)
produced a POST to "/v1", which LM Studio answers with "Unexpected endpoint
or method ... Returning 200 anyway" — a 200 with no usable body, which the
old code turned into a silent no-op translation.

The HTTP layer is exercised against a real aiohttp server on a loopback
port rather than a mock, so the request shape LM Studio actually receives
is what gets asserted.
"""
import asyncio
import json

import pytest

from aiohttp import web

from pipeline import translator as T


# ── URL normalization ────────────────────────────────────────────────

class TestHostNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("http://localhost:1234", "http://localhost:1234"),
        ("http://localhost:1234/", "http://localhost:1234"),
        ("http://localhost:1234/v1", "http://localhost:1234"),
        ("http://localhost:1234/v1/", "http://localhost:1234"),
        ("http://localhost:1234/api/v1", "http://localhost:1234"),
        ("http://localhost:1234/v1/chat/completions", "http://localhost:1234"),
        ("http://localhost:1234/api/v1/chat", "http://localhost:1234"),
        ("http://192.168.1.9:1234/v1", "http://192.168.1.9:1234"),
        ("", "http://localhost:1234"),
    ])
    def test_any_spelling_reduces_to_the_host_root(self, raw, expected):
        assert T._lm_studio_host(raw) == expected

    def test_endpoints_are_built_from_the_root(self):
        assert T.LM_STUDIO_CHAT_ENDPOINT.endswith("/api/v1/chat")
        assert T.LM_STUDIO_COMPAT_ENDPOINT.endswith("/v1/chat/completions")
        assert T.LM_STUDIO_MODELS_ENDPOINT.endswith("/v1/models")
        # The regression: never post to a bare /v1
        assert not T.LM_STUDIO_CHAT_ENDPOINT.rstrip("/").endswith("/v1")


# ── Response parsing ─────────────────────────────────────────────────

class TestNativeResponseParsing:
    def test_takes_message_content(self):
        assert T._parse_native_response(
            {"output": [{"type": "message", "content": "Привет"}]}
        ) == "Привет"

    def test_ignores_reasoning_items(self):
        """A thinking model returns reasoning as its own output item; only
        the message is the translation."""
        out = T._parse_native_response({"output": [
            {"type": "reasoning", "content": "The user wants Russian. Let me think…"},
            {"type": "message", "content": "Привет"},
        ]})
        assert out == "Привет"

    def test_concatenates_multiple_messages(self):
        assert T._parse_native_response({"output": [
            {"type": "message", "content": "Привет "},
            {"type": "message", "content": "мир"},
        ]}) == "Привет мир"

    def test_reasoning_only_raises_actionable_error(self):
        """Budget exhausted by thinking — the old code returned '' here and
        the segment silently kept its English text."""
        with pytest.raises(T.LMStudioError, match="only reasoning"):
            T._parse_native_response({
                "output": [{"type": "reasoning", "content": "hmm " * 100}],
                "stats": {"reasoning_output_tokens": 500, "total_output_tokens": 500},
            })

    def test_empty_output_raises(self):
        with pytest.raises(T.LMStudioError):
            T._parse_native_response({"output": []})

    def test_tolerates_nested_content_lists(self):
        assert T._parse_native_response({"output": [
            {"type": "message", "content": [{"type": "text", "content": "Привет"}]},
        ]}) == "Привет"


class TestCompatResponseParsing:
    def test_takes_message_content(self):
        assert T._parse_compat_response(
            {"choices": [{"message": {"content": " Привет "}}]}
        ) == "Привет"

    def test_strips_inline_think_block(self):
        out = T._parse_compat_response({"choices": [{"message": {
            "content": "<think>Russian for hello is Привет</think>Привет"}}]})
        assert out == "Привет"

    def test_missing_choices_raises(self):
        """This is literally what a POST to /v1 returned."""
        with pytest.raises(T.LMStudioError, match="no 'choices'"):
            T._parse_compat_response({"error": "Unexpected endpoint or method."})

    def test_reasoning_only_raises(self):
        with pytest.raises(T.LMStudioError, match="only reasoning"):
            T._parse_compat_response({"choices": [{"message": {
                "content": "", "reasoning_content": "thinking..."}}]})


class TestErrorEnvelope:
    def test_structured_error(self):
        err = T._lm_studio_error(json.dumps({"error": {
            "message": "Invalid model identifier", "code": "model_not_found",
            "param": "model"}}))
        assert err["code"] == "model_not_found"

    def test_bare_string_error(self):
        err = T._lm_studio_error('{"error":"Unexpected endpoint or method. (POST /v1)"}')
        assert "Unexpected endpoint" in err["message"]

    def test_non_json_body(self):
        assert T._lm_studio_error("<html>502</html>") == {}


class TestCleanTranslation:
    @pytest.mark.parametrize("raw,expected", [
        ("Translation: Привет", "Привет"),
        ("  Привет  ", "Привет"),
        ('"Привет"', "Привет"),
        ("«Привет»", "Привет"),
        ("<think>reasoning</think>Привет", "Привет"),
        ("Перевод: Привет", "Привет"),
    ])
    def test_strips_scaffolding(self, raw, expected):
        assert T._clean_translation(raw) == expected

    def test_keeps_inner_quotes(self):
        assert T._clean_translation('Он сказал "да" вчера') == 'Он сказал "да" вчера'


# ── HTTP behaviour against a stub LM Studio ──────────────────────────

class StubLMStudio:
    """Minimal stand-in for LM Studio; records the requests it receives."""

    def __init__(self):
        self.requests = []          # (path, body)
        self.native_status = 200
        self.native_body = {"output": [{"type": "message", "content": "Привет"}]}
        self.compat_body = {"choices": [{"message": {"content": "Привет (compat)"}}]}

    async def _native(self, request):
        body = await request.json()
        self.requests.append(("/api/v1/chat", body))
        if callable(self.native_body):
            status, payload = self.native_body(body)
        else:
            status, payload = self.native_status, self.native_body
        return web.json_response(payload, status=status)

    async def _compat(self, request):
        self.requests.append(("/v1/chat/completions", await request.json()))
        return web.json_response(self.compat_body)

    async def _catch_all(self, request):
        # Mirrors LM Studio's real behaviour for an unknown route
        self.requests.append((request.path, None))
        return web.json_response(
            {"error": f"Unexpected endpoint or method. ({request.method} {request.path})"},
            status=404,
        )

    def app(self):
        app = web.Application()
        app.router.add_post("/api/v1/chat", self._native)
        app.router.add_post("/v1/chat/completions", self._compat)
        app.router.add_route("*", "/{tail:.*}", self._catch_all)
        return app


@pytest.fixture
def stub(monkeypatch):
    """Run a stub LM Studio and point the client's endpoints at it."""
    server = StubLMStudio()
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(server.app())
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = runner.addresses[0][1]
    base = f"http://127.0.0.1:{port}"

    monkeypatch.setattr(T, "LM_STUDIO_CHAT_ENDPOINT", f"{base}/api/v1/chat")
    monkeypatch.setattr(T, "LM_STUDIO_COMPAT_ENDPOINT", f"{base}/v1/chat/completions")
    monkeypatch.setattr(T, "_LM_USE_NATIVE", None)
    monkeypatch.setattr(T, "_LM_SEND_REASONING", True)
    monkeypatch.setattr(T, "LM_STUDIO_REASONING", "off")

    server.loop = loop
    server.base = base
    yield server

    loop.run_until_complete(runner.cleanup())
    loop.close()


def _run(stub, coro):
    """Drive a client coroutine on the same loop as the stub server."""
    return stub.loop.run_until_complete(coro)


class TestChatRequest:
    def test_posts_to_the_native_endpoint(self, stub):
        out = _run(stub, T.lm_studio_chat("translate this", model="m"))
        assert out == "Привет"
        assert stub.requests[0][0] == "/api/v1/chat"

    def test_uses_the_native_request_schema(self, stub):
        _run(stub, T.lm_studio_chat("translate this", model="m",
                                    system_prompt="You translate."))
        body = stub.requests[0][1]
        # Native API: `input` + `system_prompt` + `max_output_tokens` — NOT
        # the OpenAI `messages`/`max_tokens` shape.
        assert body["input"] == "translate this"
        assert body["system_prompt"] == "You translate."
        assert "max_output_tokens" in body and "max_tokens" not in body
        assert "messages" not in body
        assert body["stream"] is False
        assert body["reasoning"] == "off"

    def test_output_budget_is_large_enough_for_thinking_models(self, stub):
        _run(stub, T.lm_studio_chat("x", model="m"))
        # The old hard-coded 500 was what a 27B thinking model blew through
        # on reasoning alone.
        assert stub.requests[0][1]["max_output_tokens"] >= 2048

    def test_falls_back_to_compat_when_native_route_is_missing(self, stub, monkeypatch):
        monkeypatch.setattr(T, "LM_STUDIO_CHAT_ENDPOINT", f"{stub.base}/api/v1/nope")
        out = _run(stub, T.lm_studio_chat("x", model="m"))
        assert out == "Привет (compat)"
        assert stub.requests[-1][0] == "/v1/chat/completions"
        body = stub.requests[-1][1]
        assert body["messages"][0]["role"] == "system" or "messages" in body

    def test_unknown_model_does_not_trigger_the_compat_fallback(self, stub):
        """LM Studio returns 404 for a bad model too — downgrading the
        endpoint for the whole run because of a typo would hide the cause."""
        stub.native_status = 404
        stub.native_body = {"error": {
            "message": 'Invalid model identifier "nope".',
            "type": "invalid_request", "param": "model", "code": "model_not_found"}}
        with pytest.raises(T.LMStudioError, match="not available in LM Studio"):
            _run(stub, T.lm_studio_chat("x", model="nope"))
        assert all(p != "/v1/chat/completions" for p, _ in stub.requests)

    def test_retries_without_reasoning_when_the_model_rejects_it(self, stub):
        def responder(body):
            if "reasoning" in body:
                return 400, {"error": {
                    "message": "Invalid enum value", "param": "reasoning",
                    "code": "invalid_enum_value"}}
            return 200, {"output": [{"type": "message", "content": "Привет"}]}
        stub.native_body = responder

        assert _run(stub, T.lm_studio_chat("x", model="m")) == "Привет"
        assert len(stub.requests) == 2
        assert "reasoning" in stub.requests[0][1]
        assert "reasoning" not in stub.requests[1][1]
        # And it stops sending the field afterwards
        assert T._LM_SEND_REASONING is False

    def test_server_error_raises_instead_of_returning_the_prompt(self, stub):
        stub.native_status = 500
        stub.native_body = {"error": {"message": "engine crashed"}}
        with pytest.raises(T.LMStudioError, match="engine crashed"):
            _run(stub, T.lm_studio_chat("x", model="m"))


class TestTranslateTextPropagatesFailures:
    def test_failure_raises_so_the_caller_can_retry(self, stub):
        """translate_segments() retries three times; that only works if
        translate_text stops swallowing the error and returning the source."""
        stub.native_status = 500
        stub.native_body = {"error": {"message": "boom"}}
        with pytest.raises(T.LMStudioError):
            _run(stub, T.translate_text("Hello", "ru", model="m"))

    def test_success_returns_the_cleaned_translation(self, stub):
        stub.native_body = {"output": [
            {"type": "message", "content": "Translation: Привет"}]}
        assert _run(stub, T.translate_text("Hello", "ru", model="m")) == "Привет"

    def test_segments_are_translated_end_to_end(self, stub):
        segs = [{"start": 0, "end": 1, "text": "Hello"},
                {"start": 1, "end": 2, "text": "World"}]
        out = _run(stub, T.translate_segments(segs, "ru", model="m", max_concurrent=1))
        assert [s["translated_text"] for s in out] == ["Привет", "Привет"]

    def test_failed_segments_fall_back_to_source_after_retries(self, stub):
        stub.native_status = 500
        stub.native_body = {"error": {"message": "boom"}}
        segs = [{"start": 0, "end": 1, "text": "Hello"}]
        out = _run(stub, T.translate_segments(segs, "ru", model="m", max_concurrent=1))
        # Still marked untranslated (== source) so the retry-stage UI can
        # offer "only failed segments" with a different model.
        assert out[0]["translated_text"] == "Hello"


# ── Thinking-model handling ──────────────────────────────────────────

class TestNoThinkSwitch:
    """qwen3.6-27b accepts `reasoning: "off"` and then thinks anyway (1606
    reasoning tokens for an 8-word sentence, measured). Qwen's in-prompt
    switch does work, so we append it for Qwen models."""

    @pytest.mark.parametrize("model,expected", [
        ("qwen/qwen3.6-27b", True),
        ("qwen/qwen3-8b", True),
        ("qwq-32b", True),
        ("google/gemma-4-e4b", False),
        ("qwen2.5:14b", False),      # not a thinking model — leave it alone
        ("", False),
    ])
    def test_only_qwen_thinking_models_get_the_suffix(self, model, expected):
        assert T._supports_no_think(model) is expected

    def test_suffix_is_appended_for_qwen(self, stub, monkeypatch):
        monkeypatch.setattr(T, "LM_STUDIO_REASONING", "off")
        _run(stub, T.lm_studio_chat("Translate: hi", model="qwen/qwen3.6-27b"))
        assert stub.requests[0][1]["input"].endswith("/no_think")

    def test_suffix_is_not_appended_for_other_models(self, stub, monkeypatch):
        monkeypatch.setattr(T, "LM_STUDIO_REASONING", "off")
        _run(stub, T.lm_studio_chat("Translate: hi", model="google/gemma-4-e4b"))
        assert "/no_think" not in stub.requests[0][1]["input"]

    def test_suffix_is_not_appended_when_reasoning_is_wanted(self, stub, monkeypatch):
        monkeypatch.setattr(T, "LM_STUDIO_REASONING", "high")
        _run(stub, T.lm_studio_chat("Translate: hi", model="qwen/qwen3.6-27b"))
        assert "/no_think" not in stub.requests[0][1]["input"]

    def test_echoed_control_token_is_stripped_from_output(self):
        assert T._clean_translation("Привет /no_think") == "Привет"

    def test_warns_once_when_a_model_ignores_reasoning_off(self, monkeypatch, caplog):
        monkeypatch.setattr(T, "LM_STUDIO_REASONING", "off")
        monkeypatch.setattr(T, "_LM_WARNED_IGNORED_REASONING", False)
        payload = {
            "output": [{"type": "reasoning", "content": "…"},
                       {"type": "message", "content": "Привет"}],
            "stats": {"reasoning_output_tokens": 1606, "total_output_tokens": 1620},
        }
        with caplog.at_level("WARNING"):
            for _ in range(3):
                assert T._parse_native_response(payload) == "Привет"
        warnings = [r for r in caplog.records if "ignored reasoning" in r.message]
        assert len(warnings) == 1, "should warn once, not once per segment"
