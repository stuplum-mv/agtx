pub mod hook_status;
mod operations;
pub mod spec;
pub mod trust;

pub use operations::{AgentOperations, AgentRegistry, CodingAgent, RealAgentRegistry};
pub use spec::{
    spec, AgentDialog, AgentSpec, CommandSyntax, DialogScope, HookConfigKind, HookEventSource,
    McpConfigKind, PromptForm, ResumeArgs, SendStrategy, SkillLayout, AGENT_SPECS,
};

#[cfg(feature = "test-mocks")]
pub use operations::{MockAgentOperations, MockAgentRegistry};

use serde::{Deserialize, Serialize};

/// Known coding agents that agtx can work with
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Configured instance identity (the registry lookup key).
    pub name: String,
    /// Built-in agent identity used to resolve declarative behavior.
    #[serde(default)]
    pub base_name: String,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    pub command: String,
    pub description: String,
    pub co_author: String,
}

/// How an agent accepts an initial prompt at launch.
///
/// `Argv`/`FlagInteractive` let agtx hand the opening message to the process
/// itself, so there is no window in which keystrokes can be dropped and nothing
/// to poll for. `Unknown` keeps the historical path: launch bare, wait for the
/// TUI to look ready, then type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptInjection {
    /// Positional argument: `claude "<text>"`.
    Argv,
    /// Interactive prompt flag: `gemini -i "<text>"`.
    FlagInteractive(&'static str),
    /// No verified interactive launch form — send after readiness instead.
    Unknown,
}

impl Agent {
    /// Whether this agent can take the opening message at launch.
    ///
    /// Derived from the agent's spec: the launch *form* is always known, but it
    /// is only used once verified against the real binary, because getting this
    /// wrong means the task text is silently swallowed (a `-p`-style flag that
    /// runs headless and exits) rather than failing loudly.
    pub fn prompt_injection(&self) -> PromptInjection {
        match spec::spec(self.base_name()) {
            Some(s) if s.launch_prompt_verified => match s.prompt_form {
                spec::PromptForm::Argv => PromptInjection::Argv,
                spec::PromptForm::Flag(flag) => PromptInjection::FlagInteractive(flag),
            },
            _ => PromptInjection::Unknown,
        }
    }

    pub fn new(name: &str, command: &str, description: &str, co_author: &str) -> Self {
        Self {
            name: name.to_string(),
            base_name: name.to_string(),
            profile: None,
            model: None,
            command: command.to_string(),
            description: description.to_string(),
            co_author: co_author.to_string(),
        }
    }

    /// Create a configured instance while retaining the base agent's behavior.
    pub fn named(name: &str, base: &Agent, profile: Option<String>, model: Option<String>) -> Self {
        Self {
            name: name.to_string(),
            base_name: base.base_name().to_string(),
            profile,
            model,
            command: base.command.clone(),
            description: base.description.clone(),
            co_author: base.co_author.clone(),
        }
    }

    /// Built-in identity used for specs, distinct from the configured instance name.
    pub fn base_name(&self) -> &str {
        if self.base_name.is_empty() {
            &self.name
        } else {
            &self.base_name
        }
    }

    fn push_selection_args<'a>(&'a self, spec: &spec::AgentSpec, args: &mut Vec<&'a str>) {
        if let (Some(flag), Some(value)) = (spec.profile_flag, self.profile.as_deref()) {
            args.push(flag);
            args.push(value);
        }
        if let (Some(flag), Some(value)) = (spec.model_flag, self.model.as_deref()) {
            args.push(flag);
            args.push(value);
        }
    }

    /// Argv for the agent's headless (print) invocation, used for one-shot
    /// generation like PR descriptions.
    ///
    /// Note this deliberately does *not* apply the interactive launch env
    /// (Gemini's `GEMINI_TRUST_WORKSPACE`): headless runs never touch the
    /// workspace, and adding it here would change behaviour.
    pub fn headless_invocation<'a>(&'a self, prompt: &'a str) -> (&'a str, Vec<&'a str>) {
        match spec::spec(self.base_name()) {
            Some(s) => {
                let mut args = Vec::new();
                self.push_selection_args(s, &mut args);
                args.extend_from_slice(s.headless_args);
                args.push(prompt);
                (s.binary, args)
            }
            None => (self.command.as_str(), vec![prompt]),
        }
    }

    /// Check if this agent is installed on the system
    pub fn is_available(&self) -> bool {
        which::which(&self.command).is_ok()
    }

    /// Build the shell command to resume the agent's most recent session
    /// in the current working directory. Used to recover from tmux/server restarts.
    pub fn build_resume_command(&self) -> String {
        let Some(s) = spec::spec(self.base_name()) else {
            // Nothing is known about how this agent resumes; start it fresh.
            return self.build_interactive_command("");
        };
        let resume_args: Vec<&str> = match s.resume {
            spec::ResumeArgs::Append(extra) => s.base_args.iter().chain(extra).copied().collect(),
            spec::ResumeArgs::Replace(args) => args.to_vec(),
        };
        spec::compose_command_with_options(
            s,
            &resume_args,
            None,
            self.profile.as_deref(),
            self.model.as_deref(),
        )
    }

    /// Build the shell command to start the agent interactively.
    /// When prompt is empty, the agent starts with no initial message
    /// (task content and skill commands are sent later via tmux send_keys).
    pub fn build_interactive_command(&self, prompt: &str) -> String {
        match spec::spec(self.base_name()) {
            Some(s) => spec::compose_command_with_options(
                s,
                s.base_args,
                Some(prompt),
                self.profile.as_deref(),
                self.model.as_deref(),
            ),
            None if prompt.is_empty() => self.command.clone(),
            None => format!("{} '{}'", self.command, prompt.replace('\'', "'\"'\"'")),
        }
    }
}

/// Get the list of known agents, in preference order.
///
/// Derived from [`AGENT_SPECS`] — adding an agent means adding a spec entry,
/// not editing this function.
pub fn known_agents() -> Vec<Agent> {
    spec::AGENT_SPECS
        .iter()
        .map(|s| Agent::new(s.name, s.binary, s.description, s.co_author))
        .collect()
}

/// Detect which agents are available on the system
pub fn detect_available_agents() -> Vec<Agent> {
    known_agents()
        .into_iter()
        .filter(|a| a.is_available())
        .collect()
}

/// Get a specific agent by name
pub fn get_agent(name: &str) -> Option<Agent> {
    known_agents().into_iter().find(|a| a.name == name)
}

/// Agent availability status for display
#[derive(Debug)]
pub struct AgentStatus {
    pub agent: Agent,
    pub available: bool,
}

/// Get status of all known agents
pub fn all_agent_status() -> Vec<AgentStatus> {
    known_agents()
        .into_iter()
        .map(|agent| {
            let available = agent.is_available();
            AgentStatus { agent, available }
        })
        .collect()
}

/// Parse user input for agent selection.
/// Returns the index (0-based) of the selected agent, or None for invalid input.
/// Empty input returns Some(0) (first agent as default).
pub fn parse_agent_selection(input: &str, agent_count: usize) -> Option<usize> {
    let input = input.trim();
    if input.is_empty() {
        return Some(0);
    }
    if let Ok(num) = input.parse::<usize>() {
        if num >= 1 && num <= agent_count {
            return Some(num - 1);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_omp_commands_quote_selection_and_prompt() {
        let base = get_agent("omp").unwrap();
        let agent = Agent::named(
            "omp-work",
            &base,
            Some("team's profile".into()),
            Some("openai/gpt 5".into()),
        );

        assert_eq!(agent.name, "omp-work");
        assert_eq!(agent.base_name(), "omp");
        assert_eq!(
            agent.build_interactive_command("fix 'this'"),
            "omp --profile 'team'\"'\"'s profile' --model 'openai/gpt 5' --auto-approve --plugin-dir .agtx/omp-plugin 'fix '\"'\"'this'\"'\"''"
        );
        assert_eq!(
            agent.build_resume_command(),
            "omp --profile 'team'\"'\"'s profile' --model 'openai/gpt 5' --auto-approve --plugin-dir .agtx/omp-plugin --continue"
        );

        let (binary, args) = agent.headless_invocation("describe");
        assert_eq!(binary, "omp");
        assert_eq!(
            args,
            vec![
                "--profile",
                "team's profile",
                "--model",
                "openai/gpt 5",
                "--auto-approve",
                "--no-session",
                "--print",
                "describe"
            ]
        );
    }
}
