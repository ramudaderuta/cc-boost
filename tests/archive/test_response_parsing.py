"""Unit tests for response parsing functionality."""

import pytest
from unittest.mock import MagicMock
from src.core.boost_model_manager import BoostModelManager
from src.core.config import Config


class TestResponseParsing:
    """Test response parsing accuracy with various formats."""

    @pytest.fixture
    def boost_manager(self):
        """Create a BoostModelManager instance for testing."""
        config = MagicMock()
        config.boost_base_url = "https://api.test.com/v1"
        config.boost_api_key = "sk-test-key"
        config.boost_model = "gpt-4o"
        config.boost_wrapper_template = None
        config.request_timeout = 30
        return BoostModelManager(config)

    def test_extract_section_with_complex_content(self, boost_manager):
        """Test extracting section with complex multi-line content."""
        text = """
ANALYSIS:
The user wants to analyze market trends from sales data.
Initial context: We need to locate and process sales data from 2024.
First uncertainty: Exact location and format of the sales data file.
Potential path 1: Check common data directories.
Potential path 2: Search for files with sales-related names.
Refining thought: Path 1 is better as it follows standard conventions.
After identifying the file, we'll need to parse the data and create visualizations.

GUIDANCE:
1. Call read_file with path: '/data/sales_2024.csv'
2. Call bash with command: 'python analyze_sales.py /data/sales_2024.csv'
3. Call write_file with path: '/reports/sales_analysis.html' and content: Generate comprehensive HTML report with charts

---
Additional context
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        guidance = boost_manager._extract_section(text, "GUIDANCE:")

        expected_analysis = """The user wants to analyze market trends from sales data.
Initial context: We need to locate and process sales data from 2024.
First uncertainty: Exact location and format of the sales data file.
Potential path 1: Check common data directories.
Potential path 2: Search for files with sales-related names.
Refining thought: Path 1 is better as it follows standard conventions.
After identifying the file, we'll need to parse the data and create visualizations."""

        expected_guidance = """1. Call read_file with path: '/data/sales_2024.csv'
2. Call bash with command: 'python analyze_sales.py /data/sales_2024.csv'
3. Call write_file with path: '/reports/sales_analysis.html' and content: Generate comprehensive HTML report with charts"""

        assert analysis == expected_analysis
        assert guidance == expected_guidance

    def test_extract_section_with_empty_lines(self, boost_manager):
        """Test extracting section with empty lines."""
        text = """
ANALYSIS:

The user wants to debug an API endpoint.

First, check logs.

Then examine code.

GUIDANCE:
1. Call read_file with path: '/logs/error.log'
2. Call bash with command: 'grep -i error /logs/error.log'
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        assert analysis == "The user wants to debug an API endpoint.\n\nFirst, check logs.\n\nThen examine code."

    def test_extract_section_case_insensitive_headers(self, boost_manager):
        """Test extracting section with different header formatting."""
        text = """
analysis:
This should still work even with lowercase header.

guidance:
1. Call test_tool
---
"""

        # Should not match case-insensitive headers
        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        guidance = boost_manager._extract_section(text, "GUIDANCE:")

        assert analysis is None
        assert guidance is None

    def test_extract_section_with_special_characters(self, boost_manager):
        """Test extracting section with special characters."""
        text = """
ANALYSIS:
The user wants to process JSON data with special chars: { "key": "value", "array": [1, 2, 3] }
File path contains special chars: /tmp/test file (1).json

GUIDANCE:
1. Call read_file with path: '/tmp/test file (1).json'
2. Call bash with command: 'python process_json.py "/tmp/test file (1).json"'
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        guidance = boost_manager._extract_section(text, "GUIDANCE:")

        assert '{ "key": "value", "array": [1, 2, 3] }' in analysis
        assert '/tmp/test file (1).json' in analysis
        assert 'python process_json.py "/tmp/test file (1).json"' in guidance

    def test_extract_section_multiple_occurrences(self, boost_manager):
        """Test extracting section when header appears multiple times."""
        text = """
ANALYSIS:
First analysis section.

GUIDANCE:
First guidance.

ANALYSIS:
Second analysis section - this should be ignored.

GUIDANCE:
Second guidance section - this should be ignored.
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        guidance = boost_manager._extract_section(text, "GUIDANCE:")

        # Should extract the first occurrence
        assert analysis == "First analysis section."
        assert guidance == "First guidance."

    def test_extract_section_with_colons_in_content(self, boost_manager):
        """Test extracting section when content contains colons."""
        text = """
ANALYSIS:
The user wants to process data with timestamps: 2024-01-01T12:00:00Z
Error message: Failed to connect: Connection refused
Status: HTTP 500 Internal Server Error

GUIDANCE:
1. Call bash with command: 'curl -X GET http://api.example.com/data'
2. Call read_file with path: '/config/settings.json'
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        guidance = boost_manager._extract_section(text, "GUIDANCE:")

        assert "2024-01-01T12:00:00Z" in analysis
        assert "Connection refused" in analysis
        assert "HTTP 500 Internal Server Error" in analysis
        assert "curl -X GET http://api.example.com/data" in guidance

    def test_extract_section_unicode_content(self, boost_manager):
        """Test extracting section with unicode characters."""
        text = """
ANALYSIS:
The user wants to process international data: café, naïve, résumé
File names: /tmp/测试文件.txt, /tmp/данные.csv

GUIDANCE:
1. Call read_file with path: '/tmp/测试文件.txt'
2. Call bash with command: 'python process.py --encoding utf-8'
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        guidance = boost_manager._extract_section(text, "GUIDANCE:")

        assert "café, naïve, résumé" in analysis
        assert "/tmp/测试文件.txt" in analysis
        assert "/tmp/данные.csv" in analysis
        assert "python process.py --encoding utf-8" in guidance

    def test_extract_section_very_long_content(self, boost_manager):
        """Test extracting section with very long content."""
        long_content = "\n".join([f"Line {i}: This is a test line with some content." for i in range(1000)])

        text = f"""
ANALYSIS:
{long_content}

GUIDANCE:
1. Call read_file with path: '/large/file.txt'
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")

        # Should handle long content
        assert len(analysis) > 40000  # Should be quite long
        assert "Line 0: This is a test line" in analysis
        assert "Line 999: This is a test line" in analysis

    def test_extract_section_with_code_blocks(self, boost_manager):
        """Test extracting section with code blocks."""
        text = """
ANALYSIS:
The user wants to generate Python code for data processing.

```python
import pandas as pd
import matplotlib.pyplot as plt

def process_data(file_path):
    df = pd.read_csv(file_path)
    return df.describe()
```

This code will read and analyze the data.

GUIDANCE:
1. Call write_file with path: '/scripts/process_data.py' and content: '''import pandas as pd
def process_data(file_path):
    df = pd.read_csv(file_path)
    return df.describe()'''
2. Call bash with command: 'python /scripts/process_data.py /data/input.csv'
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        guidance = boost_manager._extract_section(text, "GUIDANCE:")

        assert "import pandas as pd" in analysis
        assert "def process_data(file_path):" in analysis
        assert "import pandas as pd" in guidance
        assert "python /scripts/process_data.py /data/input.csv" in guidance

    def test_extract_section_with_nested_structure(self, boost_manager):
        """Test extracting section with nested/multi-level structure."""
        text = """
ANALYSIS:
The user wants to create a complex data processing pipeline.

Requirements:
- Input: CSV files from /data/input/
- Processing: Clean, transform, aggregate
- Output: HTML reports to /reports/

Steps:
1. Validate input files exist
2. Process each file individually
3. Combine results
4. Generate visualizations

Potential issues:
- Missing files
- Invalid data formats
- Memory constraints

GUIDANCE:
1. Call bash with command: 'find /data/input -name "*.csv" -type f'
2. Call read_file with path: '/config/pipeline_config.json'
3. Call bash with command: 'python run_pipeline.py --input /data/input --output /reports'
4. Call write_file with path: '/reports/summary.html' and content: Generate HTML summary with charts
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        guidance = boost_manager._extract_section(text, "GUIDANCE:")

        assert "Requirements:" in analysis
        assert "Potential issues:" in analysis
        assert "find /data/input -name \"*.csv\" -type f" in guidance
        assert "python run_pipeline.py --input /data/input --output /reports" in guidance

    def test_extract_section_edge_empty_section(self, boost_manager):
        """Test extracting section that exists but is empty."""
        text = """
ANALYSIS:

GUIDANCE:
1. Call test_tool
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        assert analysis is None

    def test_extract_section_edge_section_at_end(self, boost_manager):
        """Test extracting section at the end of text."""
        text = """
Some introductory text.

ANALYSIS:
This is the final section with no terminator after it.
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        assert analysis == "This is the final section with no terminator after it."

    def test_extract_section_edge_multiple_separators(self, boost_manager):
        """Test extracting section with multiple separator lines."""
        text = """
ANALYSIS:
Content before separator.

---
Content after separator.

---
More content.

GUIDANCE:
1. Call test_tool
---
"""

        analysis = boost_manager._extract_section(text, "ANALYSIS:")
        assert analysis == "Content before separator."
