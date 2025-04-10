import pytest
from unittest.mock import patch, MagicMock
import os
import json
from dotenv import load_dotenv

# Import the modules to test
from intura_ai.experiments import ChatModelExperiment
from intura_ai.platform import DashboardPlatform
from intura_ai.platform.domain import ExperimentModel, ExperimentTreatmentModel

# Test constants
MOCK_EXPERIMENT_ID = "exp_12345"
MOCK_RESPONSE = {
    "content": "Your daily motivation: Success is not final, failure is not fatal: it is the courage to continue that counts.",
    "model": "claude-3-5-sonnet-20240620",
    "provider": "Anthropic"
}

class TestDashboardPlatform:
    """Tests for the DashboardPlatform class"""
    
    def test_init(self):
        """Test initialization of DashboardPlatform"""
        client = DashboardPlatform()
        assert isinstance(client, DashboardPlatform)
    
    @patch('intura_ai.platform.DashboardPlatform.create_experiment')
    def test_create_experiment(self, mock_create_experiment):
        """Test creating an experiment"""
        # Setup mock
        mock_create_experiment.return_value = MOCK_EXPERIMENT_ID
        
        # Create client and experiment model
        client = DashboardPlatform()
        experiment = ExperimentModel(
            experiment_name="Test Experiment",
            treatment_list=[
                ExperimentTreatmentModel(
                    treatment_model_name="model1",
                    treatment_model_provider="Provider1",
                    prompt="Test prompt"
                )
            ]
        )
        
        # Call method
        experiment_id = client.create_experiment(experiment)
        
        # Assertions
        assert experiment_id == MOCK_EXPERIMENT_ID
        mock_create_experiment.assert_called_once_with(experiment)


class TestChatModelExperiment:
    """Tests for the ChatModelExperiment class"""
    
    def test_init(self):
        """Test initialization of ChatModelExperiment"""
        chat_client = ChatModelExperiment()
        assert isinstance(chat_client, ChatModelExperiment)
    
    @patch('intura_ai.experiments.ChatModelExperiment.build')
    def test_build(self, mock_build):
        """Test building a chain"""
        # Setup mock
        mock_chain = MagicMock()
        mock_build.return_value = mock_chain
        
        # Create client
        chat_client = ChatModelExperiment()
        
        # Test data
        experiment_id = MOCK_EXPERIMENT_ID
        features = {
            "user_id": "Test123",
            "feature_x1": "FREE",
            "feature_x2": "PART_TIME"
        }
        messages = [{
            "role": "human",
            "content": "test message"
        }]
        
        # Call method
        chain = chat_client.build(
            experiment_id=experiment_id,
            features=features,
            messages=messages
        )
        
        # Assertions
        assert chain == mock_chain
        mock_build.assert_called_once_with(
            experiment_id=experiment_id,
            features=features,
            messages=messages
        )
    
    @patch('intura_ai.experiments.ChatModelExperiment.build')
    def test_chain_invoke(self, mock_build):
        """Test invoking the chain"""
        # Setup mock chain with invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MOCK_RESPONSE
        mock_build.return_value = mock_chain
        
        # Create client
        chat_client = ChatModelExperiment()
        
        # Build chain
        chain = chat_client.build(
            experiment_id=MOCK_EXPERIMENT_ID,
            features={"user_id": "Test123"},
            messages=[{"role": "human", "content": "test"}]
        )
        
        # Invoke chain
        result = chain.invoke({})
        
        # Assertions
        assert result == MOCK_RESPONSE
        mock_chain.invoke.assert_called_once_with({})


@pytest.fixture
def mock_env_variables():
    """Fixture to set up environment variables for testing"""
    with patch.dict(os.environ, {
        "INTURA_API_KEY": os.environ.get("INTURA_API_KEY"),
        "INTURA_API_BASE_URL": "https://test-api.example.com"
    }):
        yield


class TestIntegration:
    """Integration tests for the experiment module"""
    
    @patch('intura_ai.platform.DashboardPlatform.create_experiment')
    @patch('intura_ai.experiments.ChatModelExperiment.build')
    def test_full_workflow(self, mock_build, mock_create_experiment, mock_env_variables):
        """Test the full workflow from creating an experiment to getting results"""
        # Setup mocks
        mock_create_experiment.return_value = MOCK_EXPERIMENT_ID
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MOCK_RESPONSE
        mock_build.return_value = mock_chain
        
        # Create clients
        dashboard_client = DashboardPlatform()
        chat_client = ChatModelExperiment()
        
        # Create experiment
        experiment = ExperimentModel(
            experiment_name="Integration Test Experiment",
            treatment_list=[
                ExperimentTreatmentModel(
                    treatment_model_name="gemini-1.5-flash",
                    treatment_model_provider="Google",
                    prompt="Act as personal assistant"
                ),
                ExperimentTreatmentModel(
                    treatment_model_name="claude-3-5-sonnet-20240620",
                    treatment_model_provider="Anthropic",
                    prompt="Act as personal assistant"
                )
            ]
        )
        
        experiment_id = dashboard_client.create_experiment(experiment)
        assert experiment_id == MOCK_EXPERIMENT_ID
        
        # Build and invoke chain
        features = {
            "user_id": "TestUser123",
            "feature_x1": "FREE",
            "feature_x2": "FULL_TIME"
        }
        
        messages = [{
            "role": "human",
            "content": "give me today motivation"
        }]
        
        chain = chat_client.build(
            experiment_id=experiment_id,
            features=features,
            messages=messages
        )
        
        result = chain.invoke({})
        
        # Assertions
        assert result == MOCK_RESPONSE
        assert "content" in result
        assert "model" in result
        assert "provider" in result


class TestExperimentModel:
    """Tests for the ExperimentModel and ExperimentTreatmentModel classes"""
    
    def test_experiment_model_creation(self):
        """Test creation of ExperimentModel"""
        treatment = ExperimentTreatmentModel(
            treatment_model_name="test-model",
            treatment_model_provider="Test Provider",
            prompt="Test prompt"
        )
        
        experiment = ExperimentModel(
            experiment_name="Test Experiment",
            treatment_list=[treatment]
        )
        
        assert experiment.experiment_name == "Test Experiment"
        assert len(experiment.treatment_list) == 1
        assert experiment.treatment_list[0] == treatment
    
    def test_treatment_model_creation(self):
        """Test creation of ExperimentTreatmentModel"""
        treatment = ExperimentTreatmentModel(
            treatment_model_name="test-model",
            treatment_model_provider="Test Provider",
            prompt="Test prompt"
        )
        
        assert treatment.treatment_model_name == "test-model"
        assert treatment.treatment_model_provider == "Test Provider"
        assert treatment.prompt == "Test prompt"


if __name__ == "__main__":
    pytest.main(["-v"])