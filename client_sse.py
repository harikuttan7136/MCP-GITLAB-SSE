import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any
from mcp import ClientSession
from mcp.client.sse import sse_client


from dotenv import load_dotenv
import boto3

load_dotenv()
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')

def to_bedrock_format(tools_list: List[Dict]) -> List[Dict]:
    return [{
        "toolSpec": {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": tool["input_schema"]["properties"],
                    "required": tool["input_schema"].get("required", [])
                }
            }
        }
    } for tool in tools_list]


def _make_bedrock_request(messages: List[Dict], tools: List[Dict]) -> Dict:
    return bedrock.converse(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig={"maxTokens": 1000, "temperature": 0},
        toolConfig={"tools": tools}
    )



class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()


    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()
        await self.session.initialize()
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"text":query}]
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        bedrock_tools = to_bedrock_format(available_tools)
        response = _make_bedrock_request(messages, bedrock_tools)

        final_text = []
        if response['stopReason'] == 'tool_use':
            final_text.append("received toolUse request")
            for item in response['output']['message']['content']:
                if 'text' in item:
                    final_text.append(f"/n[Thinking: {item['text']}]")
                elif 'toolUse' in item:
                    tool_info = item['toolUse']
                    tool_name = tool_info['name']
                    tool_args = tool_info['input']

                    final_text.append(f"/n[Calling tool {tool_name} with args {tool_args}]")
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append({"call": tool_name, "result": result})

        messages.append({
                    "role": "user", 
                    "content": [{"text": str(final_text)}]
                })
              
        response = _make_bedrock_request(messages, bedrock_tools)
        return response['output']['message']['content'][0]['text']
  

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
