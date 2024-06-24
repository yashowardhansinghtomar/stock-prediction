import streamlit as st
import yfinance as yf
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
import pandas as pd

# Custom Langchain tool to store data
class LangchainTool:
  def __init__(self, name, data):
    self.name = name
    self.data = data

  def bind(self, function_name, function):
    setattr(self, function_name, function)

# Function to fetch stock data
def fetch_stock_data(ticker, start, end):
  stock = yf.Ticker(ticker)
  data = stock.history(start=start, end=end)
  return data

# Function to set up retrieval chain from fetched stock data
def setup_retrieval_chain_from_stock_data(stock_data):
  # Create a Langchain tool to store the stock data
  stock_data_tool = LangchainTool(name="stock_data_tool", data=stock_data)

  # This tool simply retrieves the data stored earlier
  def retrieve_stock_data(context):
    return context["stock_data"]

  # Bind the retrieval function to the tool
  stock_data_tool.bind("retrieve_stock_data", retrieve_stock_data)

  # Return the Langchain tool for use in the retrieval chain
  return stock_data_tool

# Initialize LLM (ChatGroq)
llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key="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")

# Streamlit app
def main():
  # Streamlit app layout
  st.title('Indian Stock Market Data Fetcher and Q&A')

  # Input for stock ticker
  market = st.selectbox('Select Market', ['NSE', 'BSE'])
  ticker = st.text_input('Enter Stock Ticker', 'RELIANCE')

  # Input for start and end dates
  start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
  end_date = st.date_input('End Date', pd.to_datetime('today'))

  # Fetch and display data
  fetch_data_button = st.button('Fetch Data', key='fetch_data_button')

  if fetch_data_button:
    if ticker:
      full_ticker = f"{ticker}.NS" if market == 'NSE' else f"{ticker}.BO"
      try:
        data = fetch_stock_data(full_ticker, start_date, end_date)
        if not data.empty:
          st.write(f'Stock data for {ticker} on {market}')
          st.line_chart(data['Close'])
          st.dataframe(data)

          # Setup retrieval chain from fetched stock data
          retrieval_chain = setup_retrieval_chain_from_stock_data(data)

          # Interactive section with LLM
          st.header('Ask Questions to the Language Model')

          user_question = st.text_input('Ask a question:', key='user_question_input')
          get_answer_button = st.button('Get Answer', key='get_answer_button')

          if get_answer_button and user_question:
            try:
              # Agent setup
              agent = create_tool_calling_agent(llm, [retrieval_chain])
              agent_executor = AgentExecutor(agent=agent, tools=[retrieval_chain], verbose=True)

              # Accessing stock data from within the retrieval function
              def retrieve_stock_data(context):
                return context["stock_data_tool"].data  # Get data from the tool

              # Add the custom retrieval function to the agent's context
              agent.context["stock_data"] = retrieve_stock_data

              # Executing the agent with user input question
              response = agent_executor.invoke({"input": user_question})
              st.text(f"Input: {response['input']}")
              st.text(f"Output: {response['output']}")
            except Exception as e:
              st.error(f"Error generating response: {e}")
          elif get_answer_button and not user_question:
            st.error('Please enter a question.')
        else:
          st.error('No data found. Please check the ticker symbol and try again.')
      except Exception as e:
        st.error(f"Error fetching data: {e}")
  else:
    st.error('Please enter a valid stock ticker.')

if __name__ == '__main__':
  main()