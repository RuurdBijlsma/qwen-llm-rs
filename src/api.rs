use std::io::Write;
use async_stream::try_stream;
use base64::{engine::general_purpose, Engine as _};
use color_eyre::Result;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::pin::Pin;
use std::time::Instant;
use tracing::info;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessagePart>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum MessagePart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone)]
pub enum ChatEvent {
    Content(String),
    Reasoning(String),
}

#[derive(Clone)]
pub struct LlamaClient {
    http: reqwest::Client,
    base_url: String,
}

impl LlamaClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    pub async fn full_request(
        &self,
        model: String,
        messages: Vec<Message>,
    ) -> Result<ChatFullResponse> {
        let request = ChatRequest {
            model,
            messages,
            temperature: 0.7,
            stream: false,
            reasoning_format: "deepseek".to_string(),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.http.post(url).json(&request).send().await?;

        if !response.status().is_success() {
            let status =response.status();
            let err = response.text().await?;
            return Err(color_eyre::eyre::eyre!("API Error ({}): {}",status , err));
        }

        Ok(response.json().await?)
    }

    pub async fn stream_request(
        &self,
        model: String,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let request = ChatRequest {
            model,
            messages,
            temperature: 0.7,
            stream: true,
            reasoning_format: "deepseek".to_string(),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.http.post(url).json(&request).send().await?;

        if !response.status().is_success() {
            return Err(color_eyre::eyre::eyre!("API Error: {}", response.status()));
        }

        let mut stream_bytes = response.bytes_stream();
        Ok(Box::pin(try_stream! {
            while let Some(item) = stream_bytes.next().await {
                let chunk_bytes = item?;
                let text = String::from_utf8_lossy(&chunk_bytes);
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() || line == "data: [DONE]" { continue; }
                    if let Some(data) = line.strip_prefix("data: ") {
                        if let Ok(chunk) = serde_json::from_str::<ChatChunk>(data) {
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(r) = &choice.delta.reasoning_content { yield ChatEvent::Reasoning(r.clone()); }
                                if let Some(c) = &choice.delta.content { yield ChatEvent::Content(c.clone()); }
                            }
                        }
                    }
                }
            }
        }))
    }
}

pub struct ChatSession {
    client: LlamaClient,
    pub model: String,
    pub messages: Vec<Message>,
}

impl ChatSession {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: LlamaClient::new(base_url),
            model: model.to_string(),
            messages: Vec::new(),
        }
    }

    pub async fn chat<P: AsRef<Path> + Sync>(
        &mut self,
        prompt: &str,
        images: &[P],
    ) -> Result<String> {
        self.prepare_user_message(prompt, images)?;
        let response = self.client.full_request(self.model.clone(), self.messages.clone()).await?;
        let choice = response.choices.first()
            .ok_or_else(|| color_eyre::eyre::eyre!("No choices in response"))?;
        let content = choice.message.content.clone().unwrap_or_default();
        self.push_text("assistant", content.clone());
        Ok(content)
    }

    pub async fn chat_stream<P: AsRef<Path> + Sync>(
        &mut self,
        prompt: &str,
        image_paths: &[P],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        self.prepare_user_message(prompt, image_paths)?;
        self.client.stream_request(self.model.clone(), self.messages.clone()).await
    }

    fn prepare_user_message<P: AsRef<Path> + Sync>(&mut self, prompt: &str, images: &[P]) -> Result<()> {
        let mut parts = vec![MessagePart::Text { text: prompt.to_string() }];
        for path in images {
            let bytes = std::fs::read(path)?;
            let mime_type = infer::get(&bytes).map_or("image/jpeg", |kind| kind.mime_type());
            let b64 = general_purpose::STANDARD.encode(&bytes);
            let data_url = format!("data:{mime_type};base64,{b64}");
            parts.push(MessagePart::ImageUrl {
                image_url: ImageUrl { url: data_url },
            });
        }
        self.messages.push(Message {
            role: "user".to_string(),
            content: MessageContent::Parts(parts),
        });
        Ok(())
    }

    pub fn push_text(&mut self, role: &str, text: String) {
        self.messages.push(Message {
            role: role.to_string(),
            content: MessageContent::Text(text),
        });
    }

    pub fn reset(&mut self) { self.messages.clear(); }
    pub fn set_model(&mut self, model: &str) { self.model = model.to_string(); }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    stream: bool,
    reasoning_format: String,
}

#[derive(Deserialize)]
struct ChatChunk { choices: Vec<ChunkChoice> }
#[derive(Deserialize)]
struct ChunkChoice { delta: ChunkDelta }
#[derive(Deserialize)]
struct ChunkDelta { content: Option<String>, reasoning_content: Option<String> }

#[derive(Deserialize)]
pub struct ChatFullResponse {
    pub choices: Vec<FullChoice>,
}
#[derive(Deserialize)]
pub struct FullChoice {
    pub message: FullMessage,
}
#[derive(Deserialize)]
pub struct FullMessage {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
}

pub async fn run() -> Result<()> {
    let mut session = ChatSession::new("http://localhost:8080", "");

    let img_island = Path::new("assets/img/island.png");
    let img_farm = Path::new("assets/img/farm.png");
    let img_torus = Path::new("assets/img/torus.png");
    let prompt = "Caption this image in one paragraph. Respond with the caption only.";

    let now = Instant::now();

    // Warmup
    info!("Island: {}", session.chat(prompt, &[img_island]).await?);
    session.reset();

    let now2 = Instant::now();
    info!("Farm: {}", session.chat(prompt, &[img_farm]).await?);
    session.reset();
    info!("Torus: {}", session.chat(prompt, &[img_torus]).await?);
    session.reset();
    info!("Island again: {}", session.chat(prompt, &[img_island]).await?);
    info!("Follow up: {}", session.chat("Where might this be?", &[] as &[&Path]).await?);
    info!("Similarities: {}", session.chat("What are the similarities with this picture?", &[img_torus]).await?);

    info!("Total time for [API]: {:?}", now.elapsed());
    info!("Total time for [API] (excluding warmup): {:?}", now2.elapsed());

    let mut stream = session.chat_stream("How do I install minecraft?", &[] as &[&Path]).await?;
    let mut full_response = String::new();
    while let Some(event) = stream.next().await {
        if let ChatEvent::Content(c) = event? {
            print!("{c}");
            std::io::stdout().flush()?;
            full_response.push_str(&c);
        }
    }
    println!();
    session.push_text("assistant", full_response);

    Ok(())
}