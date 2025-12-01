from google.adk.agents import BaseAgent # Core ADK classes
from google.adk.events import Event # Correct Event import for ADK
from google.adk.agents.invocation_context import InvocationContext # For state and context
from google.genai.types import Content, Part # For creating proper content
import pandas as pd # For data manipulation
from typing import AsyncGenerator # For async generator type hint

class DataCollectorAgent(BaseAgent):
    # _run_async_impl is the heart of a BaseAgent's execution logic
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_name = self.name # Accessing the agent's configured name
        print(f"[{agent_name}]: Collecting data...")
        try:
            # In a real scenario, this could be an API call, database query, etc.
            # For this tutorial, we load from a predefined CSV file.
            df = pd.read_csv("data/sample_sales_data.csv")
            
            # Store the collected data (as a JSON string for easy serialization) into the shared session state.
            # This makes it accessible to the next agent in the pipeline.
            ctx.session.state["raw_data_json"] = df.to_json(orient="records")
            
            # Create proper Event with Content
            content = Content(parts=[Part(text="Data collection complete. Raw data loaded and stored in state.")])
            yield Event(content=content, author=agent_name)
        except FileNotFoundError:
            error_msg = f"Error: sample_sales_data.csv not found. Make sure it's in the 'data' directory."
            print(f"[{agent_name}]: {error_msg}")
            content = Content(parts=[Part(text=error_msg)])
            yield Event(content=content, author=agent_name, turn_complete=True)
        except Exception as e:
            error_msg = f"Error during data collection: {str(e)}"
            print(f"[{agent_name}]: {error_msg}")
            content = Content(parts=[Part(text=error_msg)])
            yield Event(content=content, author=agent_name, turn_complete=True) 