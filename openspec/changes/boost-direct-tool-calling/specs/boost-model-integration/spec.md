# Boost Model Integration

## ADDED Requirements

### Requirement: Boost Model Configuration
The system SHALL support configuration of a separate boost model API for planning and guidance generation.

#### Scenario:
User sets BOOST_BASE_URL, BOOST_API_KEY, and BOOST_MODEL in environment. When starting the proxy, it validates the boost model configuration and displays boost model status alongside existing model mappings.

#### Scenario:
User omits boost configuration. Proxy starts normally with boost features disabled, logging that boost mode is not configured.

#### Scenario:
Boost is disabled for a specific model tier. Requests for that tier use direct proxy path with standard tool-calling, no loop mechanism involved.

### Requirement: Tools-in-Message Construction
When boost mode is enabled for a request, the system SHALL construct a message that embeds tool definitions in the content (not in the tools parameter).

#### Scenario:
Claude Code sends a request with tools for file operations. The boost model receives a message containing the ANALYSIS/GUIDANCE format instructions, tool definitions embedded as text, and the user's request to "analyze market trends from data files".

#### Scenario:
Request contains no tools. Boost model receives a message with empty tools section, instructing it to provide ANALYSIS and GUIDANCE based on its knowledge.

#### Scenario:
System sends request to boost model without a tools parameter, only with tool descriptions in the message content to ensure compatibility with providers that disable tools.

### Requirement: Response Section Parsing
The system SHALL parse boost model responses into ANALYSIS, GUIDANCE, and SUMMARY sections using string pattern matching.

#### Scenario:
Boost model responds with "ANALYSIS:\nThe user requests sales data analysis. Initial context: We need to locate and process sales data from 2024. First uncertainty: Exact location and format of the sales data file. Potential path: Check common data directories. Refining thought: Start with '/data/sales_2024.csv' as it's the standard location. If not found, we can search for alternative locations. The analysis will require reading the data, processing it, and creating visualizations.\n\nGUIDANCE:\n1. Call read_file with path: '/data/sales_2024.csv'\n2. Create a visualization using the data\n3. Call write_file with path: '/reports/sales_analysis.html' and content: Generate an HTML report with charts showing monthly trends". System extracts each section correctly.

#### Scenario:
Boost model provides only GUIDANCE section: "GUIDANCE:\nCall search_web with query: 'latest Node.js performance optimizations' then call write_file with path: '/notes/nodejs_perf.md' and content: Document the key performance tips found". System gracefully handles missing ANALYSIS section and proceeds with GUIDANCE only.

#### Scenario:
Boost model provides detailed sequential analysis: "ANALYSIS:\nUser wants to debug a failing API endpoint. Initial context: The endpoint returns 500 errors. First uncertainty: Is this a code issue or configuration problem? Potential path 1: Check logs first to identify the error pattern. Potential path 2: Examine the code directly. Refining thought: Logs should be checked first as they provide immediate error context. After identifying the error, we can trace through the code. Second uncertainty: Whether the database connection is working. This will need to be verified after the initial error identification. The approach is sequential: logs → code review → database check."

#### Scenario:
Boost model uses different formatting (e.g., "Instructions:" instead of "GUIDANCE:"). System uses flexible pattern matching to still identify sections.

#### Scenario:
Boost model responds with "SUMMARY:\nThe answer to your question is 42. No tools needed for this query." System detects SUMMARY section and returns it directly without calling auxiliary model.

#### Scenario:
Boost model provides all three sections. System prioritizes SUMMARY section and returns it directly, ignoring ANALYSIS and GUIDANCE.

### Requirement: Auxiliary Model Request Assembly
When no SUMMARY section is present, the system SHALL combine the original request, boost model's ANALYSIS and GUIDANCE sections, and proper tool definitions into a request for the auxiliary model.

#### Scenario:
After boost model provides detailed analysis and guidance, system creates auxiliary request with: original user message, boost ANALYSIS providing sequential reasoning and context tracing as background, boost GUIDANCE containing specific tool calls and operations (e.g., "1. Call read_file with path: '/config/settings.json' 2. Call write_file with path: '/config/backup.json' and content: [exact content to write]"), and tools parameter with actual tool definitions.

#### Scenario:
Boost model provided only GUIDANCE with detailed instructions: "GUIDANCE:\nExecute these steps: 1) Call bash with command: 'git status' 2) Call bash with command: 'git add .' 3) Call bash with command: 'git commit -m \"Auto commit\"'". System includes these detailed instructions in auxiliary request and proceeds without additional context.

#### Scenario:
Boost model provided SUMMARY section. System skips auxiliary model assembly and returns SUMMARY directly.

### Requirement: Tool Parameter Restoration
The system SHALL restore the tools parameter when sending requests to the auxiliary model.

#### Scenario:
Boost model receives tools in message content (no tools parameter). Auxiliary model receives the same tools in the proper tools parameter for actual execution.

#### Scenario:
System ensures that the auxiliary model can execute tools while the boost model only receives tool information as text.

### Requirement: Boost Model Fallback
If the boost model call fails or response is unparsable, the system SHALL gracefully fall back to direct execution without breaking the request.

#### Scenario:
Boost API returns 500 error. System logs the failure and proceeds with direct proxy execution using the executor model directly.

#### Scenario:
Boost model response cannot be parsed into required sections. System falls back to direct execution with appropriate logging.

#### Scenario:
Boost model exceeds timeout. System aborts boost call and continues with standard tool execution path.

### Requirement: Loop Mechanism Implementation
The system SHALL implement an iterative refinement loop with a maximum of 3 iterations to improve response quality.

#### Scenario:
Boost model returns response without SUMMARY or GUIDANCE sections. System increments LOOP counter from 0 to 1 and retries with refined context including previous response.

#### Scenario:
Auxiliary model receives GUIDANCE but doesn't call any tools. System increments LOOP counter and retries boost call with context that tools weren't used.

#### Scenario:
Loop counter reaches 3 (LOOP >= 3). System exits loop and returns best available response to user, preventing infinite loops.

#### Scenario:
Each loop iteration includes previous responses and current LOOP count in the boost model context to avoid repetition and enable refinement.

### Requirement: Tool Usage Detection
The system SHALL detect whether the auxiliary model actually uses tools when executing GUIDANCE.

#### Scenario:
Auxiliary model receives GUIDANCE to call read_file but instead provides a direct answer. System detects no tools were called and increments loop for retry.

#### Scenario:
Auxiliary model calls multiple tools as guided. System detects tool usage and processes results normally without loop increment.

#### Scenario:
System monitors tool execution at the streaming level to detect usage as early as possible.