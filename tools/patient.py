from llm import llm
from graph import graph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about patients info only.
Convert the user's question based on the schema.

Use the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Example Cypher Statements:

1.Retrieve All Properties of a Patient by Patient ID
This query retrieves all properties of a patient node using their unique Patient_ID. It outputs the entire set of attributes related to the patient, including personal and medical information, enabling a comprehensive overview of their profile.
```
MATCH (p:patient {{Patient_ID: 154}})
RETURN properties(p) AS PatientProperties; 
```

2. To find all Properties of a Patient by Identity Number

```
MATCH (p:patient {{Identity_number: "950314-12-1234"}})
RETURN properties(p) AS PatientProperties; 
```



Remember before display the response, you need to list the details in bulletpoint to make it more readable.

Schema:
{schema}

Question:
{question}
"""

patient_cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

patient_cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    return_intermediate_steps=True,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=patient_cypher_prompt
)