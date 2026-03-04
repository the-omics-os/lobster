"""Test data_expert prompt includes filesystem tool instructions."""


class TestDataExpertPrompt:
    def test_prompt_includes_filesystem_section(self):
        from lobster.agents.data_expert.prompts import create_data_expert_prompt

        prompt = create_data_expert_prompt()
        assert "Filesystem" in prompt or "File Tools" in prompt
        assert "list_files" in prompt
        assert "read_file" in prompt
        assert "shell_execute" in prompt

    def test_prompt_includes_two_layer_guidance(self):
        from lobster.agents.data_expert.prompts import create_data_expert_prompt

        prompt = create_data_expert_prompt()
        # Should explain when to use file tools vs modality tools
        assert "inspect" in prompt.lower() or "investigate" in prompt.lower()

    def test_prompt_includes_file_investigation_tree(self):
        from lobster.agents.data_expert.prompts import create_data_expert_prompt

        prompt = create_data_expert_prompt()
        assert "File Investigation" in prompt

    def test_prompt_tool_count_updated(self):
        from lobster.agents.data_expert.prompts import create_data_expert_prompt

        prompt = create_data_expert_prompt()
        assert "16 specialized tools" in prompt
