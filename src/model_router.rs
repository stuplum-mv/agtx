//! Generic phase-entry model routing.
//!
//! The TUI writes a self-contained request before launching an agent. The
//! `agtx model-route` fast path resolves that request in the launch shell,
//! persists the result, scrubs the task prompt, and prints only a concrete
//! harness model id to stdout.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    Quick,
    #[default]
    Standard,
    High,
    Premium,
}

impl ModelTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Standard => "standard",
            Self::High => "high",
            Self::Premium => "premium",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Quick => 0,
            Self::Standard => 1,
            Self::High => 2,
            Self::Premium => 3,
        }
    }

    pub fn at_least(self, minimum: Self) -> Self {
        if self.index() < minimum.index() {
            minimum
        } else {
            self
        }
    }
}

fn default_router() -> String {
    "jev".to_string()
}

fn default_endpoint() -> String {
    "https://api.typesafe.ai/v1/systemone".to_string()
}

fn default_jev_model() -> String {
    "jev-latest".to_string()
}

fn default_api_key_env() -> String {
    "TYPESAFE_API_KEY".to_string()
}

fn default_timeout_ms() -> u64 {
    3_500
}

fn default_confidence_threshold() -> f64 {
    0.34
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRoutingConfig {
    #[serde(default = "default_router")]
    pub router: String,
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_jev_model")]
    pub jev_model: String,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
    #[serde(default)]
    pub phases: BTreeMap<String, PhaseModelRoutingConfig>,
    #[serde(default)]
    pub models: BTreeMap<String, AgentModelRoutes>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseModelRoutingConfig {
    #[serde(default)]
    pub fallback: ModelTier,
    #[serde(default)]
    pub minimum: ModelTier,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentModelRoutes {
    pub quick: Option<String>,
    pub standard: Option<String>,
    pub high: Option<String>,
    pub premium: Option<String>,
}

impl AgentModelRoutes {
    pub fn model_for(&self, tier: ModelTier) -> Option<&str> {
        match tier {
            ModelTier::Quick => self.quick.as_deref(),
            ModelTier::Standard => self.standard.as_deref(),
            ModelTier::High => self.high.as_deref(),
            ModelTier::Premium => self.premium.as_deref(),
        }
    }
}

impl ModelRoutingConfig {
    pub fn phase(&self, phase: &str) -> Option<&PhaseModelRoutingConfig> {
        self.phases.get(canonical_phase(phase))
    }

    pub fn routes_for(&self, base_agent: &str) -> Option<&AgentModelRoutes> {
        self.models.get(base_agent)
    }
}

pub fn canonical_phase(phase: &str) -> &str {
    match phase {
        "preresearch" => "research",
        "planning_with_research" => "planning",
        "running_with_research_or_planning" => "running",
        other => other,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRouteRequest {
    pub prompt: Option<String>,
    pub phase: String,
    pub base_agent: String,
    pub fallback_model: String,
    pub routes: AgentModelRoutes,
    pub api_key_env: String,
    pub endpoint: String,
    pub jev_model: String,
    pub timeout_ms: u64,
    pub confidence_threshold: f64,
    pub minimum: ModelTier,
    pub fallback: ModelTier,
    #[serde(default)]
    pub decision: Option<ModelRouteDecision>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRouteDecision {
    pub tier: ModelTier,
    pub model: String,
    pub confidence: Option<f64>,
    pub used_fallback: bool,
}

#[derive(Debug, Deserialize)]
struct JevResponse {
    answers: JevAnswers,
}

#[derive(Debug, Deserialize)]
struct JevAnswers {
    model_tier: JevChoice,
}

#[derive(Debug, Deserialize)]
struct JevChoice {
    choice: ModelTier,
    confidence: f64,
}

pub fn parse_jev_answer(body: &str) -> Result<(ModelTier, f64)> {
    let response: JevResponse =
        serde_json::from_str(body).context("Jev returned an invalid response")?;
    Ok((
        response.answers.model_tier.choice,
        response.answers.model_tier.confidence,
    ))
}

pub fn choose_model(
    request: &ModelRouteRequest,
    answer: Option<(ModelTier, f64)>,
) -> ModelRouteDecision {
    let fallback_tier = request.fallback.at_least(request.minimum);
    let fallback_model = request
        .routes
        .model_for(fallback_tier)
        .unwrap_or(&request.fallback_model);

    let Some((tier, confidence)) = answer else {
        return ModelRouteDecision {
            tier: fallback_tier,
            model: fallback_model.to_string(),
            confidence: None,
            used_fallback: true,
        };
    };

    let tier = tier.at_least(request.minimum);
    if confidence < request.confidence_threshold {
        return ModelRouteDecision {
            tier: fallback_tier,
            model: fallback_model.to_string(),
            confidence: Some(confidence),
            used_fallback: true,
        };
    }

    match request.routes.model_for(tier) {
        Some(model) => ModelRouteDecision {
            tier,
            model: model.to_string(),
            confidence: Some(confidence),
            used_fallback: false,
        },
        None => ModelRouteDecision {
            tier: fallback_tier,
            model: fallback_model.to_string(),
            confidence: Some(confidence),
            used_fallback: true,
        },
    }
}

fn jev_payload(request: &ModelRouteRequest) -> serde_json::Value {
    serde_json::json!({
        "state": {
            "request": request.prompt.as_deref().unwrap_or_default(),
            "phase": request.phase,
            "agent": request.base_agent,
        },
        "model": request.jev_model,
        "questions": {
            "model_tier": {
                "type": "choice",
                "instructions": "Choose the least expensive capability tier likely to complete this coding task reliably.",
                "criteria": {
                    "quick": "Small, mechanical, low-risk task with an obvious solution.",
                    "standard": "Normal implementation or debugging with limited scope.",
                    "high": "Complex reasoning, cross-cutting implementation, or consequential review.",
                    "premium": "Exceptionally difficult, ambiguous, or high-risk work requiring maximum capability."
                }
            }
        }
    })
}

fn call_jev(request: &ModelRouteRequest) -> Result<(ModelTier, f64)> {
    let api_key = std::env::var(&request.api_key_env)
        .with_context(|| format!("{} is not set", request.api_key_env))?;
    if api_key.contains(['\r', '\n']) {
        anyhow::bail!("{} contains an invalid newline", request.api_key_env);
    }

    let payload_path =
        std::env::temp_dir().join(format!("agtx-model-route-{}.json", uuid::Uuid::new_v4()));
    let mut payload =
        create_owner_only(&payload_path).context("failed to create temporary Jev payload")?;
    serde_json::to_writer(&mut payload, &jev_payload(request))?;
    drop(payload);
    let _payload_guard = RemoveOnDrop(payload_path.clone());
    let timeout = format!("{:.3}", request.timeout_ms.max(1) as f64 / 1_000.0);

    let mut last_error = None;
    for attempt in 0..2 {
        match run_curl(request, &api_key, &payload_path, &timeout) {
            Ok((status, _)) if (status == 429 || status == 529) && attempt == 0 => {
                std::thread::sleep(Duration::from_millis(150));
            }
            Ok((status, body)) => {
                if !(200..300).contains(&status) {
                    anyhow::bail!("Jev returned HTTP {status}");
                }
                return parse_jev_answer(&body);
            }
            Err(error) if attempt == 0 => {
                last_error = Some(error);
                std::thread::sleep(Duration::from_millis(150));
            }
            Err(error) => return Err(error).context("Jev request failed"),
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Jev request failed")))
}

struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn run_curl(
    request: &ModelRouteRequest,
    api_key: &str,
    payload_path: &Path,
    timeout: &str,
) -> Result<(u16, String)> {
    let mut child = Command::new("curl")
        .args([
            "--silent",
            "--show-error",
            "--max-time",
            timeout,
            "--request",
            "POST",
            "--header",
            "@-",
            "--data-binary",
            &format!("@{}", payload_path.display()),
            "--write-out",
            "\n__AGTX_HTTP_STATUS__:%{http_code}",
            "--url",
            &request.endpoint,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to run curl (is it installed?)")?;

    let mut stdin = child.stdin.take().context("failed to open curl stdin")?;
    writeln!(stdin, "Authorization: Bearer {api_key}")?;
    writeln!(stdin, "Content-Type: application/json")?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .context("failed to wait for curl")?;
    if !output.status.success() {
        anyhow::bail!(
            "curl failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let output = String::from_utf8(output.stdout).context("Jev response was not UTF-8")?;
    let (body, status) = output
        .rsplit_once("\n__AGTX_HTTP_STATUS__:")
        .context("curl response did not include an HTTP status")?;
    Ok((
        status
            .parse()
            .context("curl returned an invalid HTTP status")?,
        body.to_string(),
    ))
}

fn save_request(path: &Path, request: &ModelRouteRequest) -> Result<()> {
    let parent = path.parent().context("model route path has no parent")?;
    std::fs::create_dir_all(parent)?;
    set_directory_owner_only(parent)?;
    let temp = path.with_extension("json.tmp");
    let mut file = create_owner_only(&temp)?;
    serde_json::to_writer_pretty(&mut file, request)?;
    file.flush()?;
    drop(file);
    std::fs::rename(temp, path)?;
    Ok(())
}

fn create_owner_only(path: &Path) -> Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

#[cfg(unix)]
fn set_directory_owner_only(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_directory_owner_only(_path: &Path) -> Result<()> {
    Ok(())
}

pub fn write_request(path: &Path, request: &ModelRouteRequest) -> Result<()> {
    save_request(path, request)
}

fn resolve_request_with<F>(path: &Path, route: F) -> Result<ModelRouteDecision>
where
    F: FnOnce(&ModelRouteRequest) -> Result<(ModelTier, f64)>,
{
    let mut request: ModelRouteRequest = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;

    if let Some(decision) = request.decision.clone() {
        return Ok(decision);
    }

    let answer = match route(&request) {
        Ok(answer) => Some(answer),
        Err(error) => {
            eprintln!("agtx: model router unavailable, using fallback: {error:#}");
            None
        }
    };
    let decision = choose_model(&request, answer);
    request.prompt = None;
    request.decision = Some(decision.clone());
    save_request(path, &request)?;
    Ok(decision)
}

pub fn resolve_request(path: &Path) -> Result<ModelRouteDecision> {
    resolve_request_with(path, call_jev)
}

/// Fast-path CLI used inside agent launch command substitutions. It must emit
/// exactly one model id on stdout even when the request cannot be read.
pub fn run_cli(args: &[String]) -> Result<()> {
    let path = args.first().map(PathBuf::from);
    let fallback = args.get(1).cloned().unwrap_or_default();
    let model = path
        .as_deref()
        .and_then(|path| match resolve_request(path) {
            Ok(decision) => Some(decision.model),
            Err(error) => {
                eprintln!("agtx: model routing failed, using fallback: {error:#}");
                None
            }
        })
        .unwrap_or(fallback);
    println!("{model}");
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

/// One shell word whose value is the selected model. Every interpolated value
/// is quoted here; command composition treats the returned word as trusted.
pub fn dynamic_model_word(agtx_bin: &Path, request_path: &Path, fallback_model: &str) -> String {
    format!(
        "\"$({} model-route {} {})\"",
        shell_quote(&agtx_bin.to_string_lossy()),
        shell_quote(&request_path.to_string_lossy()),
        shell_quote(fallback_model),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(path_prompt: Option<&str>) -> ModelRouteRequest {
        ModelRouteRequest {
            prompt: path_prompt.map(str::to_string),
            phase: "planning".into(),
            base_agent: "claude".into(),
            fallback_model: "sonnet".into(),
            routes: AgentModelRoutes {
                quick: Some("haiku".into()),
                standard: Some("sonnet".into()),
                high: Some("opus".into()),
                premium: Some("opus-max".into()),
            },
            api_key_env: "MISSING_TEST_KEY".into(),
            endpoint: "https://example.invalid".into(),
            jev_model: "jev-latest".into(),
            timeout_ms: 1,
            confidence_threshold: 0.5,
            minimum: ModelTier::Standard,
            fallback: ModelTier::Standard,
            decision: None,
        }
    }

    #[test]
    fn parses_choice_response() {
        assert_eq!(
            parse_jev_answer(r#"{"answers":{"model_tier":{"choice":"high","confidence":0.91}}}"#)
                .unwrap(),
            (ModelTier::High, 0.91)
        );
    }

    #[test]
    fn minimum_and_low_confidence_fallback_are_applied() {
        let request = request(Some("task"));
        let decision = choose_model(&request, Some((ModelTier::Quick, 0.9)));
        assert_eq!(decision.tier, ModelTier::Standard);
        assert_eq!(decision.model, "sonnet");
        assert!(!decision.used_fallback);

        let decision = choose_model(&request, Some((ModelTier::Premium, 0.2)));
        assert_eq!(decision.tier, ModelTier::Standard);
        assert_eq!(decision.model, "sonnet");
        assert!(decision.used_fallback);
    }

    #[test]
    fn errors_persist_fallback_and_scrub_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("route.json");
        write_request(&path, &request(Some("secret task prompt"))).unwrap();

        let decision =
            resolve_request_with(&path, |_| anyhow::bail!("offline")).expect("fallback decision");
        assert_eq!(decision.model, "sonnet");
        assert!(decision.used_fallback);

        let stored: ModelRouteRequest =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(stored.prompt, None);
        assert_eq!(stored.decision, Some(decision));
    }

    #[test]
    fn persisted_decision_is_reused_without_calling_router() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("route.json");
        write_request(&path, &request(Some("task"))).unwrap();
        let first = resolve_request_with(&path, |_| Ok((ModelTier::High, 0.9))).unwrap();
        let second = resolve_request_with(&path, |_| panic!("router called twice")).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn dynamic_model_command_quotes_every_argument() {
        assert_eq!(
            dynamic_model_word(
                Path::new("/Applications/agtx tool"),
                Path::new("/tmp/task's route.json"),
                "provider/model one",
            ),
            "\"$('/Applications/agtx tool' model-route '/tmp/task'\"'\"'s route.json' 'provider/model one')\""
        );
    }
}
