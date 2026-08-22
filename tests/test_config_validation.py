"""Tests for settings validation and migration (app/config.py).

These settings reach a generative model, so the interesting cases are the
ones where a bad value would otherwise be persisted and silently degrade
every later dub: a string from an HTML form, an out-of-range slider, a
sentinel that must survive clamping.

The voxcpm_steps migration gets its own coverage because it changes the
meaning of a value that existing installs already have on disk.
"""
import pytest

from app import config as C


# ── coercion ────────────────────────────────────────────────────────

def test_form_strings_are_coerced_to_numbers():
    # An HTML form hands back strings; the model needs real numbers.
    assert C.coerce_field("voxcpm_cfg", "2.5") == 2.5
    assert C.coerce_field("voxcpm_steps", "12") == 12


def test_bool_accepts_checkbox_spellings():
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        assert C.coerce_field("voxcpm_denoise_refs", truthy) is True
    for falsy in ("0", "false", "no", "off", ""):
        assert C.coerce_field("voxcpm_denoise_refs", falsy) is False


def test_enum_is_trimmed_and_accepted():
    assert C.coerce_field("tts_engine", " f5tts ") == "f5tts"


def test_fields_outside_the_table_pass_through_untouched():
    # Only settings the product exposes are validated; the rest behave as before.
    assert C.coerce_field("whisper_model", "medium") == "medium"
    assert C.coerce_field("some_future_key", {"a": 1}) == {"a": 1}


# ── rejection ───────────────────────────────────────────────────────

@pytest.mark.parametrize("key,value", [
    ("voxcpm_cfg", 99),           # above range
    ("voxcpm_cfg", 0.1),          # below range
    ("voxcpm_xling_steps", 100),
    ("tts_max_stretch", 0.2),
    ("voxcpm_steps", 2),          # non-sentinel below the real minimum
])
def test_out_of_range_is_rejected_not_clamped(key, value):
    # Rejecting means the user finds out; clamping means they discover a
    # different number later and can't tell why.
    with pytest.raises(ValueError):
        C.coerce_field(key, value)


def test_unknown_enum_choice_is_rejected():
    with pytest.raises(ValueError) as e:
        C.coerce_field("tts_engine", "bogus")
    assert "voxcpm" in str(e.value)      # error names the valid choices


def test_non_numeric_is_rejected():
    with pytest.raises(ValueError):
        C.coerce_field("voxcpm_cfg", "abc")


# ── the steps sentinel ──────────────────────────────────────────────

def test_zero_survives_as_the_follow_the_tier_sentinel():
    # 0 is outside the 4–24 override range but must not be rejected: it is
    # how "no override, use the speed tier" is expressed.
    assert C.coerce_field("voxcpm_steps", 0) == 0
    assert C.coerce_field("voxcpm_steps", "0") == 0


def test_range_bounds_are_inclusive():
    assert C.coerce_field("voxcpm_steps", 4) == 4
    assert C.coerce_field("voxcpm_steps", 24) == 24
    assert C.coerce_field("voxcpm_cfg", 1.0) == 1.0
    assert C.coerce_field("voxcpm_cfg", 3.0) == 3.0


# ── update() is all-or-nothing ──────────────────────────────────────

def _detached():
    """A config that never touches the real config-user.json."""
    c = C.UserConfig()
    c._save = lambda: None
    return c


def test_update_applies_nothing_when_one_value_is_bad():
    c = _detached()
    before = (c.voxcpm_cfg, c.tts_engine)
    with pytest.raises(ValueError):
        c.update(voxcpm_cfg=2.4, tts_engine="nonsense")
    # A half-applied Settings save is worse than a rejected one.
    assert (c.voxcpm_cfg, c.tts_engine) == before


def test_update_applies_all_good_values():
    c = _detached()
    c.update(voxcpm_cfg="2.4", voxcpm_steps="16", voxcpm_denoise_refs="on")
    assert (c.voxcpm_cfg, c.voxcpm_steps, c.voxcpm_denoise_refs) == (2.4, 16, True)


def test_unknown_keys_are_ignored_not_fatal():
    c = _detached()
    c.update(definitely_not_a_setting=1, voxcpm_cfg=2.2)
    assert c.voxcpm_cfg == 2.2
    assert not hasattr(c, "definitely_not_a_setting")


# ── migration ───────────────────────────────────────────────────────

def test_legacy_dead_default_becomes_follow_the_tier():
    # voxcpm_steps defaulted to 10 while being ignored on the batch path, so
    # most existing files carry it. Honouring it now would move "balanced"
    # from 8 to 10 steps on upgrade — a behaviour change nobody asked for.
    c = _detached()
    c.voxcpm_steps = C._LEGACY_VOXCPM_STEPS
    C._migrate(c, 0)
    assert c.voxcpm_steps == 0
    assert c.config_version == C.CONFIG_VERSION


def test_migration_leaves_a_deliberate_override_alone():
    c = _detached()
    c.voxcpm_steps = 16
    C._migrate(c, 0)
    assert c.voxcpm_steps == 16


def test_migration_does_not_rerun_on_an_already_migrated_config():
    # Someone who deliberately picks 10 after migrating must keep it.
    c = _detached()
    c.voxcpm_steps = 10
    c.config_version = C.CONFIG_VERSION
    C._migrate(c, C.CONFIG_VERSION)
    assert c.voxcpm_steps == 10


def test_fresh_install_needs_no_migration():
    c = C.UserConfig()
    assert c.voxcpm_steps == 0
    assert c.config_version == C.CONFIG_VERSION


# ── the background_volume migration ─────────────────────────────────

def test_bg_ducking_accepts_checkbox_spellings():
    for truthy in ("1", "true", "yes", "on"):
        assert C.coerce_field("bg_ducking", truthy) is True
    for falsy in ("0", "false", "off", ""):
        assert C.coerce_field("bg_ducking", falsy) is False


def test_legacy_flat_mix_level_becomes_the_ducked_bed_ceiling():
    # _save() wrote 0.15 into every existing file, so a stored 0.15 means
    # "the old default" and should follow it to the new ducked-bed 0.5.
    c = _detached()
    c.background_volume = C._LEGACY_BG_VOLUME
    C._migrate(c, 1)
    assert c.background_volume == 0.5
    assert c.config_version == C.CONFIG_VERSION


def test_bg_migration_leaves_a_deliberate_level_alone():
    c = _detached()
    c.background_volume = 0.3
    C._migrate(c, 1)
    assert c.background_volume == 0.3


def test_bg_migration_does_not_rerun_on_an_already_migrated_config():
    # Someone who deliberately picks 0.15 after migrating must keep it.
    c = _detached()
    c.background_volume = C._LEGACY_BG_VOLUME
    c.config_version = C.CONFIG_VERSION
    C._migrate(c, C.CONFIG_VERSION)
    assert c.background_volume == C._LEGACY_BG_VOLUME
