from llm import llm
from graph import graph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate


CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Ensure that the generated Cypher statements are syntactically correct and relevant to the question.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Do not return entire nodes or embedding properties.

Fine Tuning:

Example Cypher Statements:

1. To find the information for a specific hospital by Hospital Name
```
MATCH (h:hospital {{Hospital_Name: Hospital Name"}})
RETURN h.Hospital_Name, h.Hospital_Contact, h.Hospital_ID, h.Specialisation_Offered, h.State
```

2. To find the hospitals based on single, multiple, or alternative specializations:
```
MATCH (h:hospital)
WHERE h.Specialisation_Offered CONTAINS "Cardiology" OR h.Specialisation_Offered CONTAINS "Neurology"
RETURN 
    h.Hospital_ID AS HospitalID, 
    h.Hospital_Name AS HospitalName, 
    h.Specialisation_Offered AS Specialisations;

```

3.  To find the hospitals based on single, multiple, or alternative specializations:
```
MATCH (h:hospital)
WHERE h.Specialisation_Offered CONTAINS "Orthopedics" AND h.Specialisation_Offered CONTAINS "Pulmonology"
RETURN 
    h.Hospital_ID AS HospitalID, 
    h.Hospital_Name AS HospitalName, 
    h.Specialisation_Offered AS Specialisations;

```

4. To find the hospitals in a specific state:
```
   MATCH (h:hospital) 
   WHERE h.State CONTAINS "Penang" 
   RETURN h.Hospital_ID, h.Hospital_Name;
```

5. To find the physicians with a specific specialization recruited by a specific Hospital
```
MATCH (h:hospital {{Hospital_Name: "Penang General Hospital"}})-[:recruit]->(p:physician)
WHERE p.Specialization = "Pulmonology"
RETURN p.Physician_Name AS PhysicianName, 
       p.License_Number AS LicenseNumber, 
       p.Physician_Contact AS PhysicianContactNumber;
```

Schema:
{schema}

Question:
{question}

Cypher Query:
"""


hospital_cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

hospital_cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    return_intermediate_steps=True,
    verbose=True,
    allow_dangerous_requests=True,
    validate_cypher=True,
    function_response_system="Response based on the Cypher Query and context",
    cypher_prompt=hospital_cypher_prompt
)