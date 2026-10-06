## Core Concepts
State: A central data structure (often a dictionary or TypedDict) shared across the application that keeps track of the current conversation, intermediate outputs, and variables.

Nodes: Functions or agents that perform work (e.g., calling an LLM, querying a database, parsing output) and update the shared State.

Edges: Rules that define how execution flows from one Node to another. These can be direct transitions or Conditional Edges (decision points based on the current State).