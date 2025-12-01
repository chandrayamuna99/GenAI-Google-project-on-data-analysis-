from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.genai.types import Content, Part # For creating proper content
import pandas as pd
from io import StringIO # For pandas JSON reading
import matplotlib.pyplot as plt
import seaborn as sns # For more aesthetic plots
import os # To ensure results directory exists and for path handling
from typing import AsyncGenerator

class VisualizationAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_name = self.name
        print(f"[{agent_name}]: Generating visualizations...")
        
        processed_data_json = ctx.session.state.get("processed_data_json")
        # Retrieve analyses, with defaults if not found
        gemini_analysis = ctx.session.state.get("gemini_analysis", "Gemini analysis not available.")
        openai_analysis = ctx.session.state.get("openai_analysis", "OpenAI analysis not available.")

        if not processed_data_json:
            error_msg = "Error: Processed data not found in state for visualization."
            print(f"[{agent_name}]: {error_msg}")
            content = Content(parts=[Part(text=error_msg)])
            yield Event(content=content, author=agent_name, turn_complete=True)
            return

        try:
            # Fix pandas warning by using StringIO
            df = pd.read_json(StringIO(processed_data_json), orient="records")
            # Ensure 'Date' is datetime for proper plotting
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Ensure the 'results' directory exists for saving plots
            os.makedirs("results", exist_ok=True)
            
            plot_paths = []

            # --- Plot 1: Sales Revenue Over Time ---
            plt.figure(figsize=(12, 7)) # Adjusted size for better readability
            sns.set_style("whitegrid") # Using Seaborn style
            sns.lineplot(data=df, x='Date', y='Revenue', hue='Product_Category', marker='o')
            plt.title(f"Sales Revenue Over Time\n(Gemini Trend Snippet: {gemini_analysis[:70]}...)")
            plt.xlabel("Date")
            plt.ylabel("Revenue ($)")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Product Category')
            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            sales_plot_path = os.path.join("results", "sales_revenue_over_time.png")
            plt.savefig(sales_plot_path)
            plt.close() # Close the plot to free memory
            plot_paths.append(sales_plot_path)
            content = Content(parts=[Part(text=f"Sales revenue plot saved to {sales_plot_path}")])
            yield Event(content=content, author=agent_name)

            # --- Plot 2: Total Revenue by Product Name ---
            if 'Product_Name' in df.columns and 'Revenue' in df.columns:
                plt.figure(figsize=(10, 7))
                product_revenue = df.groupby('Product_Name')['Revenue'].sum().sort_values(ascending=False)
                # Fix seaborn warning by properly setting hue
                sns.barplot(x=product_revenue.index, y=product_revenue.values, hue=product_revenue.index, palette="viridis", legend=False)
                plt.title(f"Total Revenue by Product\n(OpenAI Anomaly Note: {openai_analysis[:70]}...)")
                plt.xlabel("Product Name")
                plt.ylabel("Total Revenue ($)")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                product_plot_path = os.path.join("results", "product_revenue.png")
                plt.savefig(product_plot_path)
                plt.close()
                plot_paths.append(product_plot_path)
                content = Content(parts=[Part(text=f"Product revenue plot saved to {product_plot_path}")])
                yield Event(content=content, author=agent_name)
            
            # Store paths to generated visualizations in state
            ctx.session.state["visualization_paths"] = plot_paths
            print(f"[{agent_name}]: Generated {len(plot_paths)} visualizations")
            content = Content(parts=[Part(text="Visualizations generated successfully.")])
            yield Event(content=content, author=agent_name, turn_complete=True)

        except Exception as e:
            error_msg = f"Error during visualization generation: {str(e)}"
            print(f"[{agent_name}]: {error_msg}")
            content = Content(parts=[Part(text=error_msg)])
            yield Event(content=content, author=agent_name, turn_complete=True) 