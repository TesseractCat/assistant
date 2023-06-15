use serde::{Serialize, Deserialize};
use serde_json::value::Value;
use anyhow::{Result, Context};

const KEY: &str = include_str!("secret.key");

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "role", content = "content")]
#[serde(rename_all = "lowercase")]
pub enum Entry {
    System(String),
    Assistant(String),
    User(String),
}
impl Entry {
    pub fn content(&self) -> &str {
        match &self {
            Self::System(s) | Self::Assistant(s) | Self::User(s) => {
                s
            }
        }
    }
    pub fn as_table(&self) -> Option<Vec<Vec<String>>> {
        let mut content = self.content().to_string();
        content = content.split_off(content.find("|")?);
        content.truncate(content.rfind("|")? + 1);

        let rows: Vec<Vec<_>> =
            content.split("\n").enumerate()
            .filter_map(|(i, row)| {
                if i != 1 {
                    Some(row[1..(row.len()-1)].split("|")
                         .map(|e| e.trim().to_string()).collect())
                } else {
                    None
                }
            }).collect();

        Some(rows)
    }
    pub fn as_json(&self) -> Option<Value> {
        let mut content = self.content().to_string();
        let array = content.find("[").unwrap_or(usize::MAX) < content.find("{").unwrap_or(usize::MAX);

        let mut json: Value = serde_json::from_str(&if array {
            content = content.split_off(content.find("[")?);
            content.truncate(content.rfind("]")? + 1);
            content
        } else {
            content = content.split_off(content.find("{")?);
            content.truncate(content.rfind("}")? + 1);
            content
        }).ok()?;

        if json.is_array() && json.as_array().unwrap().len() == 1 {
            Some(json.get_mut(0).unwrap().take())
        } else {
            Some(json)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Chat {
    model: &'static str,
    messages: Vec<Entry>,

    #[serde(skip_serializing)]
    tokens: u64,
}
impl Chat {
    pub fn new() -> Self {
        Chat {
            model: "gpt-3.5-turbo",
            messages: Vec::new(),
            tokens: 0,
        }
    }
    pub fn push_entry(&mut self, entry: Entry) {
        self.messages.push(entry);
    }
    pub fn push_system(&mut self, message: impl AsRef<str>) {
        self.messages.push(Entry::System(message.as_ref().to_string()));
    }
    pub fn push_assistant(&mut self, message: impl AsRef<str>) {
        self.messages.push(Entry::Assistant(message.as_ref().to_string()));
    }
    pub fn push_user(&mut self, message: impl AsRef<str>) {
        self.messages.push(Entry::User(message.as_ref().to_string()));
    }
    pub fn last(&self) -> Option<&Entry> {
        self.messages.last()
    }

    pub async fn complete(&mut self) -> Result<&mut Self> {
        let client = reqwest::Client::new();
        let res = client.post("https://api.openai.com/v1/chat/completions")
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .bearer_auth(KEY)
            .json(&self)
            .send()
            .await?;
        let mut val = res.json::<Value>().await?;

        let completion: Entry = serde_json::from_value(
            val.get_mut("choices").context("No choices")?.take()
                .get_mut(0).context("0-length choices")?.take()
                .get_mut("message").context("No message")?.take()
        )?;
        let tokens_used = val.get("usage").context("No usage")?
            .get("total_tokens").context("No usage")?
            .as_u64().context("Not u64")?;

        self.tokens += tokens_used;
        self.push_entry(completion);

        Ok(self)
    }
}
