import asyncio
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['SERVER_BRAIN_VAULT']=str(ROOT/'sample_vault')
import brain_index as bi

def test_frontmatter_links_and_bounds(tmp_path):
    fm,body=bi.parse_frontmatter('---\ntitle: Demo\n---\n# Hello')
    assert fm['title']=='Demo'
    assert bi.extract_links('[[Nodes/demo-node|Node]]')[0][1]=='demo-node'
    assert all(p.resolve().is_relative_to(bi.resolve_vault().resolve()) for p in bi.iter_md_files(bi.resolve_vault()))

def test_real_retrieval():
    import server as s
    assert 'Inference' in s.vault_search('model serving latency',3)
    assert 'fictional' in s.get_note('Services/inference.md').lower()
    assert 'demo-node' in s.list_nodes()
    assert 'Backlinks' in s.get_note('Nodes/demo-node.md')
    assert 'No note matches' in s.get_note('../../outside.md')

def test_stdio_protocol():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    async def run():
        params=StdioServerParameters(command=sys.executable,args=[str(ROOT/'server.py')],env={'SERVER_BRAIN_VAULT':str(ROOT/'sample_vault')})
        async with stdio_client(params) as (r,w):
            async with ClientSession(r,w) as c:
                await c.initialize()
                assert len((await c.list_tools()).tools)==10
                result=await c.call_tool('get_service',{'name':'inference'})
                assert 'inference' in result.content[0].text.lower()
    asyncio.run(run())
