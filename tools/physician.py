from llm import llm
from graph import graph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate


CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
- Use only the provided relationship types and properties in the schema.
- Do not use any other relationship types or properties that are not provided.
- Ensure that the generated Cypher statements are syntactically correct and relevant to the question.
- Do not include any explanations or apologies in your responses.
- Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
- Do not include any text except the generated Cypher statement.

Do not return entire nodes or embedding properties.

Fine Tuning:

Example Cypher Statements:

1. To find physicians by partial or exact name match with hospital details
```
MATCH (p:physician)<-[:recruit]-(h:hospital)
WHERE p.Physician_Name CONTAINS "Lim" OR p.Physician_Name = "Dr. Physician Name"
RETURN p.Physician_Name AS Name, 
       p.Specialization AS Specialization, 
       p.License_Number AS LicenseNumber, 
       p.Physician_Contact AS PhysicianContactNumber,
       h.Hospital_Name AS HospitalName;

```

2. To find a physician by their license number and the hospital they work at
```
MATCH (p:physician)<-[:recruit]-(h:hospital)
WHERE p.License_Number = 48720
RETURN p.Physician_Name AS Name, 
       p.Specialization AS Specialization, 
       p.Physician_Contact AS PhysicianContactNumber,
       h.`Hospital Name` AS HospitalName;
```
3. 
```
MATCH (p:physician)<-[:recruit]-(h:hospital)
WHERE h.Hospital_Name IN $hospitalNames AND 
      p.Specialization IN $specializations
RETURN p.Physician_Name AS Name, 
       p.Specialization AS Specialization, 
       p.License_Number AS LicenseNumber, 
       p.Physician_Contact AS PhysicianContactNumber,
       h.Hospital_Name AS HospitalName;
```
Schema:
{schema}

Question:
{question}

Cypher Query:
"""


physician_cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

physician_cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    return_intermediate_steps=True,
    verbose=True,
    allow_dangerous_requests=True,
    validate_cypher=True,
    function_response_system="Response based on the Cypher Query and context",
    cypher_prompt=physician_cypher_prompt
)