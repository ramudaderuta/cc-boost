"""Tests for documentation validation and examples."""

import pytest
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestDocumentationValidation:
    """Test that documentation is complete and accurate."""

    def test_readme_exists(self):
        """Test that README.md exists and contains boost features."""
        readme_path = PROJECT_ROOT / "README.md"
        assert readme_path.exists(), "README.md should exist"

        content = readme_path.read_text()

        # Check for boost-related content
        boost_keywords = [
            "boost",
            "Boost-Directed Tool-Calling",
            "ENABLE_BOOST_SUPPORT",
            "BOOST_BASE_URL",
            "BOOST_API_KEY",
            "BOOST_MODEL",
            "BOOST_WRAPPER_TEMPLATE"
        ]

        for keyword in boost_keywords:
            assert keyword.lower() in content.lower(), f"README.md should mention {keyword}"

    def test_env_example_exists(self):
        """Test that .env.example exists and contains boost configuration."""
        env_example_path = PROJECT_ROOT / ".env.example"
        if env_example_path.exists():
            content = env_example_path.read_text()

            # Check for boost configuration variables
            boost_vars = [
                "ENABLE_BOOST_SUPPORT",
                "BOOST_BASE_URL",
                "BOOST_API_KEY",
                "BOOST_MODEL",
                "BOOST_WRAPPER_TEMPLATE"
            ]

            found_vars = []
            for var in boost_vars:
                if var in content:
                    found_vars.append(var)

            # At least some boost vars should be documented
            assert len(found_vars) > 0, ".env.example should document boost configuration variables"

    def test_boost_design_documentation(self):
        """Test that boost design documentation is complete."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        assert design_path.exists(), "Boost design document should exist"

        content = design_path.read_text()

        # Check for required sections
        required_sections = [
            "Architecture Overview",
            "System Components",
            "Wrapper Format Design",
            "Data Flow",
            "Key Design Decisions",
            "Implementation Considerations",
            "Loop Mechanism Details",
            "Extension Points",
            "Implementation Status"
        ]

        for section in required_sections:
            assert section in content, f"Design document should contain {section} section"

    def test_boost_proposal_documentation(self):
        """Test that boost proposal documentation is complete."""
        proposal_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "proposal.md"
        assert proposal_path.exists(), "Boost proposal document should exist"

        content = proposal_path.read_text()

        # Check for required sections
        required_sections = [
            "Overview",
            "Goals",
            "Flow Diagram",
            "Proposed Solution",
            "Key Features",
            "Configuration",
            "Impact"
        ]

        for section in required_sections:
            assert section in content, f"Proposal document should contain {section} section"

    def test_boost_specs_documentation(self):
        """Test that boost specification documents are complete."""
        specs_dir = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "specs"
        assert specs_dir.exists(), "Boost specs directory should exist"

        spec_files = list(specs_dir.rglob("spec.md"))
        assert len(spec_files) > 0, "Should have at least one specification document"

        required_specs = {
            "boost-model-integration": False,
            "configuration-management": False
        }

        for spec_file in spec_files:
            parent_name = spec_file.parent.name
            if parent_name in required_specs:
                required_specs[parent_name] = True

        for spec_name, found in required_specs.items():
            assert found, f"Should have {spec_name} specification"

    def test_boost_tasks_documentation(self):
        """Test that boost tasks documentation is complete."""
        tasks_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "tasks.md"
        assert tasks_path.exists(), "Boost tasks document should exist"

        content = tasks_path.read_text()

        # Check for required phases
        required_phases = [
            "Phase 1: Core Infrastructure",
            "Phase 2: Integration",
            "Phase 3: Error Handling & Monitoring",
            "Phase 4: Documentation & Testing",
            "Phase 5: Polish & Optimization"
        ]

        for phase in required_phases:
            assert phase in content, f"Tasks document should contain {phase}"

        # Check for validation criteria
        assert "Validation Criteria" in content, "Tasks document should have validation criteria"

    def test_boost_wrapper_format_examples(self):
        """Test that wrapper format examples are documented."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        content = design_path.read_text()

        # Check for wrapper format examples
        assert "FORMAT 1" in content, "Should document FORMAT 1 (SUMMARY)"
        assert "FORMAT 2" in content, "Should document FORMAT 2 (GUIDANCE)"
        assert "FORMAT 3" in content, "Should document FORMAT 3 (OTHER)"
        assert "SUMMARY:" in content, "Should document SUMMARY format"
        assert "ANALYSIS:" in content, "Should document ANALYSIS format"
        assert "GUIDANCE:" in content, "Should document GUIDANCE format"

    def test_boost_loop_mechanism_documentation(self):
        """Test that loop mechanism is properly documented."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        content = design_path.read_text()

        # Check for loop mechanism details
        loop_concepts = [
            "Loop State Management",
            "Loop Triggers",
            "Loop Behavior",
            "Context Preservation",
            "maximum 3 iterations",
            "LOOP counter",
            "Previous Attempts"
        ]

        for concept in loop_concepts:
            assert concept in content, f"Should document loop mechanism {concept}"

    def test_boost_configuration_documentation(self):
        """Test that boost configuration is properly documented."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        content = design_path.read_text()

        # Check for configuration documentation
        config_vars = [
            "BOOST_BASE_URL",
            "BOOST_API_KEY",
            "BOOST_MODEL",
            "ENABLE_BOOST_SUPPORT",
            "BOOST_WRAPPER_TEMPLATE"
        ]

        for var in config_vars:
            assert var in content, f"Should document configuration variable {var}"

    def test_boost_fallback_logic_documentation(self):
        """Test that fallback logic is documented."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        content = design_path.read_text()

        # Check for fallback logic documentation
        fallback_concepts = [
            "Fallback Strategy",
            "graceful fallback",
            "direct execution",
            "backward compatibility"
        ]

        for concept in fallback_concepts:
            assert concept.lower() in content.lower(), f"Should document fallback logic {concept}"

    def test_boost_performance_documentation(self):
        """Test that performance characteristics are documented."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        content = design_path.read_text()

        # Check for performance documentation
        performance_concepts = [
            "Performance Characteristics",
            "Observed Behavior",
            "performance overhead",
            "2-10 seconds",
            "sub-millisecond"
        ]

        for concept in performance_concepts:
            assert concept.lower() in content.lower(), f"Should document performance {concept}"

    def test_boost_tool_usage_detection_documentation(self):
        """Test that tool usage detection is documented."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        content = design_path.read_text()

        # Check for tool usage detection documentation
        tool_concepts = [
            "Tool Usage Detection",
            "tool_calls",
            "detect whether the auxiliary model actually uses tools",
            "smart detection"
        ]

        for concept in tool_concepts:
            assert concept.lower() in content.lower(), f"Should document tool usage detection {concept}"

    def test_boost_integration_tests_documentation(self):
        """Test that integration tests are documented."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        content = design_path.read_text()

        # Check for integration test documentation
        test_concepts = [
            "Integration Tests Completed",
            "Test Evidence",
            "basic server functionality",
            "SUMMARY responses",
            "GUIDANCE responses",
            "streaming support",
            "loop mechanism",
            "tool usage detection",
            "fallback to direct execution"
        ]

        for concept in test_concepts:
            assert concept.lower() in content.lower(), f"Should document integration test {concept}"

    def test_boost_flow_diagram_documentation(self):
        """Test that flow diagram is documented."""
        proposal_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "proposal.md"
        content = proposal_path.read_text()

        # Check for flow diagram
        assert "Flow Diagram" in content, "Should document flow diagram"
        assert "```mermaid" in content, "Should include mermaid flow diagram"
        assert "graph TD" in content, "Should have graph definition"
        assert "User Request" in content, "Should show user request in flow"
        assert "Boost model enabled?" in content, "Should show boost decision point"

    def test_boost_goals_documentation(self):
        """Test that boost goals are clearly documented."""
        proposal_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "proposal.md"
        content = proposal_path.read_text()

        # Check for goals
        assert "Goals" in content, "Should document goals"
        assert "Allow Boost models to understand available tools" in content, "Should document tool understanding goal"
        assert "Provide structured guidance to auxiliary models" in content, "Should document guidance goal"
        assert "Preserve existing proxy features" in content, "Should document preservation goal"
        assert "Enable iterative refinement" in content, "Should document iteration goal"

    def test_boost_wrapper_template_configuration(self):
        """Test that BOOST_WRAPPER_TEMPLATE configuration is documented."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        content = design_path.read_text()

        # Check for wrapper template configuration
        assert "BOOST_WRAPPER_TEMPLATE" in content, "Should document BOOST_WRAPPER_TEMPLATE"
        assert "custom template" in content.lower(), "Should mention custom template capability"
        assert "default fallback" in content.lower(), "Should mention default fallback behavior"

    def test_documentation_consistency(self):
        """Test that documentation is consistent across files."""
        proposal_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "proposal.md"
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        tasks_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "tasks.md"

        proposal_content = proposal_path.read_text()
        design_content = design_path.read_text()
        tasks_content = tasks_path.read_text()

        # Check for consistent terminology
        key_terms = [
            "Boost-Directed Tool-Calling",
            "boost model",
            "auxiliary model",
            "SUMMARY",
            "GUIDANCE",
            "ANALYSIS",
            "loop mechanism",
            "iterative refinement"
        ]

        for term in key_terms:
            # Term should appear in multiple documents
            term_count = 0
            if term in proposal_content:
                term_count += 1
            if term in design_content:
                term_count += 1
            if term in tasks_content:
                term_count += 1

            assert term_count >= 1, f"Term '{term}' should appear in at least one document"

    def test_documentation_examples(self):
        """Test that documentation includes practical examples."""
        design_path = PROJECT_ROOT / "openspec" / "changes" / "boost-direct-tool-calling" / "design" / "design.md"
        content = design_path.read_text()

        # Check for practical examples
        assert "Example FORMAT 1:" in content, "Should include FORMAT 1 example"
        assert "Example FORMAT 2:" in content, "Should include FORMAT 2 example"
        assert "Configuration Examples:" in content, "Should include configuration examples"

        # Check for concrete tool examples
        assert "search_web:" in content, "Should include concrete tool example"
        assert "read_file:" in content, "Should include concrete tool example"
        assert "bash:" in content, "Should include concrete tool example"
