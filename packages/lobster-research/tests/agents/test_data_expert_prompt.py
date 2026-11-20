"""Test data_expert prompt includes filesystem tool instructions."""


class TestDataExpertPrompt:
    def test_prompt_includes_filesystem_section(self):
        from lobster.agents.data_expert.prompts import create_data_expert_prompt

        prompt = create_data_expert_prompt()
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
        assert "FILE INVESTIGATION" in prompt

    def test_prompt_includes_decision_trees(self):
        from lobster.agents.data_expert.prompts import create_data_expert_prompt

        prompt = create_data_expert_prompt()
        assert "Decision_Trees" in prompt or "DOWNLOAD REQUEST" in prompt

    def test_prompt_acts_on_the_queue_claim_line_instead_of_waiting(self):
        from lobster.agents.data_expert.prompts import create_data_expert_prompt

        prompt = create_data_expert_prompt()
        assert "IN_PROGRESS → wait" not in prompt
        assert (
            "IN_PROGRESS → execute_download_from_queue: it restarts a download "
            "whose run has stopped; if it names a live process, report that and "
            "do not wait or poll"
        ) in prompt
