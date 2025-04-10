import pytest
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

# Since we're testing a script that would typically be run as "__main__",
# we need to import it in a way that allows testing
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import os

# Assuming the main script is in the same directory as this test file
# and named "experiment_script.py" (adjust path as needed)
script_path = os.path.join(os.path.dirname(__file__), '../examples', 'experiment_script.py')

# Load the script as a module for testing
spec = spec_from_loader("experiment_script", SourceFileLoader("experiment_script", script_path))
experiment_script = module_from_spec(spec)
spec.loader.exec_module(experiment_script)


class TestMainScript:
    """Tests for the main script functionality"""
    
    @patch('intura_ai.platform.DashboardPlatform.create_experiment')
    @patch('intura_ai.experiments.ChatModelExperiment.build')
    @patch('sys.stdout', new_callable=StringIO)
    def test_main_execution(self, mock_stdout, mock_build, mock_create_experiment):
        """Test the full script execution with mocked dependencies"""
        # Setup mocks
        mock_experiment_id = "exp_98765"
        mock_create_experiment.return_value = mock_experiment_id
        
        mock_response = {
            "content": "Your motivation for today: The only way to do great work is to love what you do.",
            "model": "gemini-1.5-flash",
            "provider": "Google"
        }
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        mock_build.return_value = mock_chain
        
        # Execute the main function (as if script was run directly)
        # We need to ensure the code is executed as "__main__"
        with patch.dict('sys.modules', {'__main__': experiment_script}):
            # We also need to patch __name__ to "__main__" inside the module
            with patch.object(experiment_script, '__name__', '__main__'):
                experiment_script.__dict__['__name__'] = '__main__'
                
                # Execute the script's code (which should run the if __name__ == "__main__" block)
                exec(compile(spec.loader.get_source(spec.name), spec.origin, 'exec'), experiment_script.__dict__)
        
        # Verify interactions
        mock_create_experiment.assert_called_once()
        mock_build.assert_called_once_with(
            experiment_id=mock_experiment_id,
            features={
                "user_id": "Rama12345", 
                "feature_x1": "FREE", 
                "feature_x2": "FULL_TIME",
                "feature_x3": "your custom features"
            },
            messages=[{
                "role": "human",
                "content": "give me today motivation"
            }]
        )
        mock_chain.invoke.assert_called_once_with({})
        
        # Verify output
        assert str(mock_response) in mock_stdout.getvalue()