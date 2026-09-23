//! Frequency to musical-note mapping (12-tone equal temperament, A4 = 440 Hz).

const NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Map `freq` Hz to `(note_name, cents_off)`.
///
/// `note_name` looks like `"A4"`; `cents_off` is in [-50, 50), positive when
/// sharp. Returns `None` for non-positive frequencies.
pub fn note_of(freq: f32) -> Option<(String, f32)> {
    if freq <= 0.0 || !freq.is_finite() {
        return None;
    }
    let midi_f = 69.0 + 12.0 * (freq / 440.0).log2();
    let midi = midi_f.round() as i32;
    let cents = 100.0 * (midi_f - midi as f32);
    let name = NAMES[midi.rem_euclid(12) as usize];
    let octave = midi.div_euclid(12) - 1;
    Some((format!("{name}{octave}"), cents))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a440_is_in_tune() {
        let (name, cents) = note_of(440.0).unwrap();
        assert_eq!(name, "A4");
        assert!(cents.abs() < 0.01, "{cents}");
    }

    #[test]
    fn semitones_and_octaves() {
        assert_eq!(note_of(466.16).unwrap().0, "A#4");
        assert_eq!(note_of(220.0).unwrap().0, "A3");
        assert_eq!(note_of(261.63).unwrap().0, "C4");
        assert_eq!(note_of(82.41).unwrap().0, "E2"); // guitar low E
    }

    #[test]
    fn cents_sign_convention() {
        let (_, sharp) = note_of(445.0).unwrap();
        let (_, flat) = note_of(435.0).unwrap();
        assert!(sharp > 15.0 && sharp < 25.0, "{sharp}");
        assert!(flat < -15.0 && flat > -25.0, "{flat}");
    }

    #[test]
    fn rejects_garbage() {
        assert!(note_of(0.0).is_none());
        assert!(note_of(-5.0).is_none());
        assert!(note_of(f32::NAN).is_none());
    }
}
