# Configuration Management

## MODIFIED Requirements

### Requirement: Extended Environment Configuration
The configuration system SHALL support new boost-related environment variables with validation.

#### Scenario:
User sets ENABLE_BOOST_SUPPORT=SMALL_MODEL. When a request comes for claude-3-haiku, it routes through boost execution path, while sonnet and op requests use direct proxy.

#### Scenario:
User sets invalid ENABLE_BOOST_SUPPORT value. Proxy fails to start with clear error message listing valid options: NONE, BIG_MODEL, MIDDLE_MODEL, SMALL_MODEL.

### Requirement: Per-Tier Boost Enablement
The system SHALL support enabling boost mode for specific model tiers independently.

#### Scenario:
ENABLE_BOOST_SUPPORT=BIG_MODEL,SMALL_MODEL. Opus and haiku requests use boost mode, while sonnet requests use direct execution.

#### Scenario:
Only small model has boost enabled. User requests sonnet with tools - system uses direct execution path without attempting boost model call.

### Requirement: Dynamic Configuration Updates
The system SHALL support runtime changes to boost configuration where applicable.

#### Scenario:
User changes BOOST_MODEL from gpt-5 to gpt-4-turbo. New requests immediately use new model without requiring proxy restart.

#### Scenario:
Boost API key is rotated. System detects change and uses new credentials for subsequent boost model calls.


## ADDED Requirements

### Requirement: Configuration Validation
The proxy SHALL validate boost configuration at startup and provide clear error messages for misconfigurations.

#### Scenario:
BOOST_BASE_URL is set but BOOST_API_KEY is missing. Proxy fails to start with message "BOOST_API_KEY is required when BOOST_BASE_URL is configured".

#### Scenario:
BOOST_MODEL is set to invalid model name. Proxy logs warning and proceeds with boost features disabled.

### Requirement: Configuration Documentation
All new configuration options SHALL be documented with examples and default values.

#### Scenario:
New user reviews .env.example file and sees clear documentation for each boost-related variable with example values for different providers.

#### Scenario:
User runs proxy with --help flag and sees boost configuration options alongside existing parameters.

### Requirement: Three-Section Format Support
The system SHALL support the three-section response format (ANALYSIS, GUIDANCE, SUMMARY) for boost model responses.

#### Scenario:
Boost model provides SUMMARY section with a direct answer. System detects SUMMARY section and returns it directly without calling auxiliary model.

#### Scenario:
Boost model provides only ANALYSIS and GUIDANCE sections. System proceeds with auxiliary model execution as usual.

#### Scenario:
Boost model provides all three sections. System prioritizes SUMMARY section and returns it directly.

### Requirement: Wrapper Template Configuration
The system SHALL support configurable wrapper templates for the boost model prompt construction.

#### Scenario:
User sets BOOST_WRAPPER_TEMPLATE to a custom template. System uses custom template instead of default when constructing boost model requests.

#### Scenario:
No BOOST_WRAPPER_TEMPLATE is provided. System uses built-in default template with three response formats and loop context.

#### Scenario:
Template includes placeholders for [loop number], [previous attempts], [user request], and [tool definitions] that are dynamically filled.