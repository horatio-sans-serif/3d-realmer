//! Ambient audio system: procedurally generated environmental sounds.
//!
//! Uses rodio for audio output. Generates ambient soundscapes based on
//! weather, time of day, and environment. No audio files required -
//! all sounds are synthesized.

use std::time::Duration;

use rodio::source::Source;
use rodio::{OutputStream, OutputStreamHandle, Sink};

use super::sky::DayPhase;
use super::weather::WeatherType;

/// Ambient audio state.
pub struct AmbientAudio {
    _stream: Option<OutputStream>,
    _handle: Option<OutputStreamHandle>,
    wind_sink: Option<Sink>,
    rain_sink: Option<Sink>,
    bird_sink: Option<Sink>,
    current_weather: WeatherType,
    current_phase: DayPhase,
    /// Master volume [0, 1].
    volume: f32,
    /// Audio enabled flag.
    enabled: bool,
}

impl AmbientAudio {
    pub fn new() -> Self {
        let (stream, handle, enabled) = match OutputStream::try_default() {
            Ok((s, h)) => (Some(s), Some(h), true),
            Err(e) => {
                log::warn!("Audio output unavailable: {}", e);
                (None, None, false)
            }
        };

        let mut audio = Self {
            _stream: stream,
            _handle: handle,
            wind_sink: None,
            rain_sink: None,
            bird_sink: None,
            current_weather: WeatherType::Clear,
            current_phase: DayPhase::Day,
            volume: 0.3,
            enabled,
        };

        if enabled {
            audio.start_ambient_layers();
        }

        audio
    }

    fn start_ambient_layers(&mut self) {
        let handle = match &self._handle {
            Some(h) => h,
            None => return,
        };

        // Wind layer: always present, volume varies with weather
        if let Ok(sink) = Sink::try_new(handle) {
            let wind = WindSource::new(0.5);
            sink.append(wind);
            sink.set_volume(self.volume * 0.3);
            self.wind_sink = Some(sink);
        }

        // Bird layer: chirps during daytime
        if let Ok(sink) = Sink::try_new(handle) {
            let birds = BirdSource::new();
            sink.append(birds);
            sink.set_volume(self.volume * 0.2);
            self.bird_sink = Some(sink);
        }
    }

    /// Update audio based on current weather and time of day.
    pub fn update(&mut self, weather: WeatherType, phase: DayPhase, wind_strength: f32) {
        if !self.enabled {
            return;
        }

        // Update wind volume based on weather and wind strength
        if let Some(ref sink) = self.wind_sink {
            let wind_vol = match weather {
                WeatherType::Rain => 0.5 + wind_strength * 0.3,
                WeatherType::Snow => 0.2 + wind_strength * 0.1,
                WeatherType::Clear => 0.1 + wind_strength * 0.2,
            };
            sink.set_volume(self.volume * wind_vol);
        }

        // Start/stop rain sound
        if weather != self.current_weather {
            match weather {
                WeatherType::Rain => {
                    if self.rain_sink.is_none() {
                        if let Some(ref handle) = self._handle {
                            if let Ok(sink) = Sink::try_new(handle) {
                                let rain = RainSource::new();
                                sink.append(rain);
                                sink.set_volume(self.volume * 0.4);
                                self.rain_sink = Some(sink);
                            }
                        }
                    }
                }
                _ => {
                    // Stop rain
                    if let Some(ref sink) = self.rain_sink {
                        sink.stop();
                    }
                    self.rain_sink = None;
                }
            }
            self.current_weather = weather;
        }

        // Birds are louder during day, silent at night
        if let Some(ref sink) = self.bird_sink {
            let bird_vol = match phase {
                DayPhase::Day => 0.3,
                DayPhase::Dawn => 0.5, // Dawn chorus
                DayPhase::Dusk => 0.15,
                DayPhase::Night => 0.0,
            };
            sink.set_volume(self.volume * bird_vol);
        }

        self.current_phase = phase;
    }

    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            if let Some(ref sink) = self.wind_sink {
                sink.set_volume(0.0);
            }
            if let Some(ref sink) = self.rain_sink {
                sink.set_volume(0.0);
            }
            if let Some(ref sink) = self.bird_sink {
                sink.set_volume(0.0);
            }
        }
    }
}

/// Procedural wind noise source (filtered noise).
struct WindSource {
    sample_rate: u32,
    phase: f64,
    intensity: f32,
    filter_state: f64,
}

impl WindSource {
    fn new(intensity: f32) -> Self {
        Self {
            sample_rate: 44100,
            phase: 0.0,
            intensity,
            filter_state: 0.0,
        }
    }
}

impl Iterator for WindSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        // Low-frequency modulated noise for wind
        self.phase += 1.0 / self.sample_rate as f64;

        // White noise
        let noise = pseudo_random_audio(self.phase) * 2.0 - 1.0;

        // Low-pass filter (simple one-pole)
        let cutoff = 0.02;
        self.filter_state += cutoff * (noise as f64 - self.filter_state);

        // Low-frequency volume modulation (gusts)
        let gust = ((self.phase * 0.3).sin() * 0.5 + 0.5) as f32;

        Some(self.filter_state as f32 * self.intensity * gust * 0.3)
    }
}

impl Source for WindSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None // Infinite
    }
}

/// Procedural rain noise source.
struct RainSource {
    sample_rate: u32,
    phase: f64,
    filter_state: f64,
}

impl RainSource {
    fn new() -> Self {
        Self {
            sample_rate: 44100,
            phase: 0.0,
            filter_state: 0.0,
        }
    }
}

impl Iterator for RainSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        self.phase += 1.0 / self.sample_rate as f64;

        // Pink-ish noise for rain
        let noise = pseudo_random_audio(self.phase * 3.7) * 2.0 - 1.0;

        // Bandpass filter (simple approximation)
        let cutoff = 0.05;
        self.filter_state += cutoff * (noise as f64 - self.filter_state);

        // Occasional droplet peaks
        let droplet_phase = (self.phase * 7.3).fract();
        let droplet = if droplet_phase < 0.001 { 0.5 } else { 0.0 };

        Some((self.filter_state as f32 * 0.2 + droplet as f32) * 0.3)
    }
}

impl Source for RainSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

/// Procedural bird chirp source.
struct BirdSource {
    sample_rate: u32,
    phase: f64,
    chirp_phase: f64,
}

impl BirdSource {
    fn new() -> Self {
        Self {
            sample_rate: 44100,
            phase: 0.0,
            chirp_phase: 0.0,
        }
    }
}

impl Iterator for BirdSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        self.phase += 1.0 / self.sample_rate as f64;
        self.chirp_phase += 1.0 / self.sample_rate as f64;

        // Bird chirps: short sine bursts at varying frequencies
        let chirp_interval = 2.5; // seconds between chirps
        let chirp_duration = 0.08;
        let t_in_interval = self.chirp_phase % chirp_interval;

        if t_in_interval < chirp_duration {
            // Active chirp
            let freq = 2000.0 + 1500.0 * (t_in_interval * 50.0).sin();
            let envelope = 1.0 - (t_in_interval / chirp_duration);
            let sample = (self.phase * freq * std::f64::consts::TAU).sin() * envelope;
            Some(sample as f32 * 0.15)
        } else if t_in_interval > chirp_interval * 0.4
            && t_in_interval < chirp_interval * 0.4 + chirp_duration * 0.7
        {
            // Second chirp (birds often chirp in pairs)
            let local_t = t_in_interval - chirp_interval * 0.4;
            let freq = 2500.0 + 1000.0 * (local_t * 60.0).sin();
            let envelope = 1.0 - (local_t / (chirp_duration * 0.7));
            let sample = (self.phase * freq * std::f64::consts::TAU).sin() * envelope;
            Some(sample as f32 * 0.12)
        } else {
            Some(0.0) // Silence between chirps
        }
    }
}

impl Source for BirdSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

/// Simple hash-based pseudo-random for audio generation.
fn pseudo_random_audio(phase: f64) -> f64 {
    let x = (phase * 12.9898 + phase * 78.233).sin() * 43758.5453;
    x.fract().abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wind_source_produces_samples() {
        let mut wind = WindSource::new(0.5);
        let samples: Vec<f32> = (0..1000).map(|_| wind.next().unwrap()).collect();
        // Should produce non-zero samples
        assert!(samples.iter().any(|s| s.abs() > 0.001));
        // Should be bounded
        assert!(samples.iter().all(|s| s.abs() < 1.0));
    }

    #[test]
    fn rain_source_produces_samples() {
        let mut rain = RainSource::new();
        let samples: Vec<f32> = (0..1000).map(|_| rain.next().unwrap()).collect();
        assert!(samples.iter().any(|s| s.abs() > 0.001));
        assert!(samples.iter().all(|s| s.abs() < 1.0));
    }

    #[test]
    fn bird_source_produces_chirps() {
        let mut birds = BirdSource::new();
        // Advance past silence to a chirp
        let samples: Vec<f32> = (0..44100 * 3).map(|_| birds.next().unwrap()).collect();
        // Should have both silence and non-silence
        let has_silence = samples.iter().any(|s| s.abs() < 0.0001);
        let has_chirps = samples.iter().any(|s| s.abs() > 0.01);
        assert!(has_silence, "should have silence between chirps");
        assert!(has_chirps, "should have chirp sounds");
    }

    #[test]
    fn pseudo_random_audio_range() {
        for i in 0..100 {
            let v = pseudo_random_audio(i as f64 * 0.01);
            assert!(v >= 0.0 && v <= 1.0, "out of range: {}", v);
        }
    }
}
