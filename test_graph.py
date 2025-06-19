from rag_module import build_graph

def test_build_graph():
    try:
        graph = build_graph()
        print("Graph built successfully.")
    except Exception as e:
        print("Failed to build graph.")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_build_graph()
