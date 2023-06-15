use std::fs::File;
use std::io::{BufWriter, Cursor, BufReader};
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use tokio::process::Command;

use regex::Regex;
use anyhow::Result;

// Use cpal for audio input, rodio for output
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, StreamConfig, SampleRate, BufferSize};
use rodio::{OutputStream, source::Source, Decoder};

use serde::{Serialize, Deserialize};
use webrtc_vad::Vad;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};
use rustpotter::{Rustpotter, RustpotterConfig, Wakeword};

mod circular_buffer;
use circular_buffer::CircularBuffer;

mod chat;
use chat::{Chat, Entry};

enum SpeakingState {
    Silent,
    Speaking,
    Pending { end: Instant }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut chat = Chat::new();
    chat.push_system(r#"
    You are a helpful audio-based assistant. You answer to 'computer' and 'peter', but your real name is 'Grenouille'.

    The user input will be based on STT (speech-to-text) audio input, and may not be completely accurate.
    If required, you can interface with a Python 3.5 interpreter to assist in answering queries.
    The python output will be shown after the response.

    Format your response as JSON, here are the possible responses:
    {
        "type": "response",
        "response": "Here is an example response."
    }
    {
        "type": "python",
        "response": "The answer to your question is: ",
        "python": "print(5 + 5)",
    }

    To review, here are the fields you can use:
    - type: Can be either 'response' or 'python'
    - response: The response as a string. Keep responses short and to the point.
    - python: If type is python, then the python command to run. Do not use any external dependencies when running python.

    Provide your answer in JSON form. Reply with only the answer in JSON form and include no other commentary:
    "#);
    chat.push_assistant(r#"{"type": "response", "response": "Alright, let's get started!"}"#);
    chat.push_user(format!(r#"{{"type": "user", "content": "{}"}}"#, "fje and the ant and joke"));
    chat.push_assistant(r#"{"type": "unclear", "response": "Sorry I'm not sure what you just said there. Can you rephrase that or provide more info?"}"#);

    println!("Setting up whisper...");

    let whisper_ctx = WhisperContext::new("../ggml-model-whisper-base.en-q5_1.bin").expect("Failed to load model");
    //let whisper_ctx = WhisperContext::new("../ggml-tiny.en-q4_0.bin").expect("Failed to load model");
    let mut whisper_state = whisper_ctx.create_state().expect("Failed to create state");

    println!("Setting up audio...");

    let host = cpal::default_host();
    let input_device = host.default_input_device().unwrap();
    let output_device = host.default_output_device().unwrap();

    // https://github.com/RustAudio/rodio/issues/330
    let (_output_stream, output_stream_handle) = OutputStream::try_from_device(&output_device).unwrap();
    let sink = rodio::Sink::try_new(&output_stream_handle).expect("Sink open failed");
    let play_file = |path: &str| {
        sink.append(Decoder::new(
            File::open(path).unwrap()
        ).expect("Failed to decode file"));
    };

    println!(" - {:?}", input_device.default_input_config());
    println!(" - {:?}", input_device.name());

    let config: StreamConfig = input_device.default_input_config()?.into();
    let channel_count = config.channels as usize;
    let sample_rate = config.sample_rate.0;

    // Buffer all audio data for the last 15 seconds
    let audio_buffer: Arc<Mutex<CircularBuffer<f32>>> = Arc::new(Mutex::new(CircularBuffer::new(sample_rate as usize * 15)));

    let vad_frame_length = (sample_rate as f32 * (10./1000.)) as usize;
    assert!(vad_frame_length == 160);
    let mut vad_buffer = [0.; (16000. * (10./1000.)) as usize];
    let mut vad_i16_buffer = [0i16; (16000. * (10./1000.)) as usize];
    let mut vad = Vad::new_with_rate_and_mode(webrtc_vad::SampleRate::Rate16kHz, webrtc_vad::VadMode::VeryAggressive);

    let mut rustpotter_config = RustpotterConfig::default();
    let mut rustpotter = Rustpotter::new(&rustpotter_config).unwrap();
    rustpotter.add_wakeword(Wakeword::new_from_sample_files(
        "computer".to_string(), Some(0.5), Some(0.15),
        (0..=4).map(|i| format!("./clips/{}.wav", i)).collect::<Vec<String>>()
    ).expect("Failed to add wakeword"));
    println!("samples per frame {:?}", rustpotter.get_samples_per_frame());

    let stream_handle = audio_buffer.clone();
    let stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            data
                .iter().cloned().enumerate()
                .filter(|(i, _)| i % channel_count == 0).map(|(_, sample)| sample) // Just grab the first channel
                .for_each(|sample| {
                    stream_handle.lock().unwrap().overwrite(sample)
                });
        },
        move |err| {
            eprintln!("Stream error: {:?}", err);
        },
        None
    )?;

    stream.play().expect("Failed to start audio input stream");
    
    let mut speaking = SpeakingState::Silent;
    let mut speaking_start = Instant::now();
    let mut detection_start = Instant::now();
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let mut audio_handle = audio_buffer.lock().unwrap();

        if audio_handle.len() > vad_frame_length && audio_handle.len() > 480 {
            let slices = audio_handle.as_slices(); // I think this works
            let (left, right) = vad_buffer.split_at_mut(vad_frame_length.saturating_sub(slices.1.len()));
            right.copy_from_slice(&slices.1[slices.1.len().saturating_sub(right.len())..]);
            left.copy_from_slice(&slices.0[slices.0.len().saturating_sub(left.len())..]);

            vad_i16_buffer = std::array::from_fn(|i| (vad_buffer[i].clamp(-1., 1.) * i16::MAX as f32) as i16);
            let voice_segment = vad.is_voice_segment(&vad_i16_buffer).expect("VAD failed");

            match speaking {
                SpeakingState::Silent => {
                    let mut rustpotter_buffer = [0.; 480];

                    let (left, right) = rustpotter_buffer.split_at_mut(480_usize.saturating_sub(slices.1.len()));
                    right.copy_from_slice(&slices.1[slices.1.len().saturating_sub(right.len())..]);
                    left.copy_from_slice(&slices.0[slices.0.len().saturating_sub(left.len())..]);

                    if let Some(detection) = rustpotter.process_f32(&rustpotter_buffer) {
                        println!("Rustpotter: {:?}", detection);
                        speaking = SpeakingState::Speaking;
                        speaking_start = Instant::now() - Duration::from_millis(2000); // Rustpotter is about 2 seconds slower than the start of the utterance
                        detection_start = Instant::now();
                    }
                },
                SpeakingState::Speaking => {
                    if !voice_segment {
                        speaking = SpeakingState::Pending { end: Instant::now() }
                    }
                },
                SpeakingState::Pending { end } => {
                    if voice_segment {
                        speaking = SpeakingState::Speaking;
                    } else {
                        if Instant::now() - end > Duration::from_millis(800) && Instant::now() - detection_start > Duration::from_millis(1500) {
                            speaking = SpeakingState::Silent;

                            play_file("./on.wav");

                            let speaking_duration = Instant::now() - speaking_start;
                            let speaking_duration_samples = (speaking_duration.as_secs_f32() * sample_rate as f32).ceil() as usize;
                            println!("Processing, spoke for {:?}", speaking_duration);
                            //stream.pause().expect("Failed to pause");

                            audio_handle.make_contiguous();
                            let speaking_slice = &audio_handle.as_slices().0[audio_handle.len().saturating_sub(speaking_duration_samples)..];

                            let mut whisper_params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                            whisper_params.set_print_progress(false);
                            whisper_params.set_suppress_non_speech_tokens(true);
                            let whisper_processing_start = Instant::now();
                            whisper_state.full(whisper_params, speaking_slice).expect("Failed to run whisper model");

                            let num_segments = whisper_state
                                .full_n_segments()
                                .expect("Failed to get whisper segment count");
                            let segments: Vec<_> = (0..num_segments).map(|i| {
                                let segment_text = whisper_state.full_get_segment_text(i).expect("Failed to get whisper segment");
                                strip_brackets(&segment_text.trim().to_lowercase())
                            }).collect();
                            
                            println!("Finished processing, took {:?} | {:?}x faster than realtime",
                                Instant::now() - whisper_processing_start,
                                speaking_duration.as_secs_f32()/(Instant::now() - whisper_processing_start).as_secs_f32(),
                            );
                            play_file("./done.wav");

                            let response = handle_prompt(&mut chat, segments).await?;
                            match response {
                                Some(r) => {
                                    println!("Response {:?}: ", r);

                                    if matches!(r.ty, ResponseType::Response) && r.response.is_some() {
                                        play_tts(&r.response.unwrap()).await;
                                    } else {
                                        play_file("./unclear.wav");
                                    }
                                },
                                None => play_file("./unclear.wav")
                            }

                            audio_handle.clear();
                            //stream.play().expect("Failed to play");
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
enum ResponseType {
    Response,
    Python,
    Unclear
}
#[derive(Serialize, Deserialize, Debug, Clone)]
struct AssistantResponse {
    #[serde(rename = "type")]
    ty: ResponseType,
    response: Option<String>,
    python: Option<String>
}

async fn handle_prompt(chat: &mut Chat, prompt: Vec<String>) -> Result<Option<AssistantResponse>> {
    let prompt = prompt.join(" ");

    println!("Handling prompt: {:?}", prompt);

    let computer_regex = Regex::new("^(computer|peter|[a-zA-Z]+ peter)")?; // Sometimes mistakes 'computer' for 'peter'
    if computer_regex.is_match(&prompt) {
        chat.push_user(format!(r#"{{"type": "user", "content": "{}"}}"#, prompt));
        chat.complete().await?;

        let json_response = chat.last().unwrap().content().to_string();
        Ok(serde_json::from_str(&json_response).ok())
    } else {
        Ok(None)
    }
}
async fn play_tts(text: &str) {
    Command::new("./mimic.exe")
        .arg("-voice").arg("kal")
        .arg("--setf").arg("duration_stretch=0.85")
        .arg("--setf").arg("int_f0_target_mean=75")
        .arg(format!(r#""{}""#, text))
        .spawn().expect("Mimic failed to start")
        .wait()
        .await
        .expect("Mimic failed to run");
}

fn strip_brackets(input: &str) -> String {
    let re = Regex::new(r"[\[\(].+?[\]\)]").expect("Invalid regex");
    re.replace_all(input, "").to_string()
}