from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.genai.types import Content, Part # For creating proper content
import pandas as pd
from io import StringIO # For pandas JSON reading
from typing import AsyncGenerator

class DataPreprocessorAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_name = self.name
        print(f"[{agent_name}]: Preprocessing data...")
        
        # Retrieve raw data from session state, put there by the DataCollectorAgent
        raw_data_json = ctx.session.state.get("raw_data_json")
        
        if not raw_data_json:
            error_msg = "Error: Raw data not found in state for preprocessing."
            print(f"[{agent_name}]: {error_msg}")
            content = Content(parts=[Part(text=error_msg)])
            yield Event(content=content, author=agent_name, turn_complete=True)
            return # Stop if critical data is missing

        try:
            # Fix pandas warning by using StringIO
            df = pd.read_json(StringIO(raw_data_json), orient="records")
            
            # Example preprocessing steps:
            # Handle missing values (proper way without warnings)
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # Ensure 'Date' column is in datetime format
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            print(f"[{agent_name}]: Processed {len(df)} rows of data")

            # Store the processed data back into session state for the next agents
            ctx.session.state["processed_data_json"] = df.to_json(orient="records")
            content = Content(parts=[Part(text="Data preprocessing complete. Processed data stored in state.")])
            yield Event(content=content, author=agent_name)
        except Exception as e:
            error_msg = f"Error during data preprocessing: {str(e)}"
            print(f"[{agent_name}]: {error_msg}")
            content = Content(parts=[Part(text=error_msg)])
            yield Event(content=content, author=agent_name, turn_complete=True) 