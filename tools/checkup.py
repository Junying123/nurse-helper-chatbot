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

1. To find checkup details for a patient Using Patient ID, Name, or Identity Number
```
MATCH (c:checkup)<-[:has]-(p:patient)
WHERE p.Patient_ID = $patientId 
   OR p.Name = $patientName
   OR p.Identity_number = $identityNumber
RETURN c.Checkup_ID AS CheckupID, 
       c.Checkup_Status AS CheckupStatus, 
       c.Admission_Date AS AdmissionDate, 
       c.Discharge_Date AS DischargeDate,
       c.`Length_of_Stay (LOS)` AS LengthofStay,
       c.Discharge_Condition AS DischargeCondition, 
       c.Diagnosis AS Diagnosis, 
       c.Medication AS Medication, 
       c.Room_Number AS RoomNumber;
```

2. To find all checkups attended by a specific physician and which patient they belong to
```
MATCH (p:patient)-[:has]->(c:checkup)<-[:attend]-(ph:physician)
WHERE ph.Physician_ID = $physicianId OR ph.Physician_Name = $physicianName
RETURN c.Checkup_ID AS CheckupID, 
       c.Admission_Type AS AdmissionType, 
       c.Checkup_Status AS CheckupStatus, 
       c.Diagnosis AS Diagnosis, 
       c.Treatment AS Treatment, 
       c.Specialisation AS Specialisation, 
       c.Patient_ID AS PatientID, 
       c.Room_Number AS RoomNumber,
       p.Name AS PatientName, 
       ph.Specialization AS PhysicianSpecialisation;
```

3. To find checkups conducted in a specific room with checkup and patient details
```
MATCH (p:patient)-[:has]->(c:checkup)
WHERE c.Room_Number = "A314"
RETURN c.Checkup_ID AS CheckupID, 
       c.Admission_Type AS AdmissionType, 
       c.Checkup_Status AS CheckupStatus, 
       c.Diagnosis AS Diagnosis, 
       c.Medication AS Medication,  
       p.Identity_number AS IdentityNumber, 
       p.Name AS PatientName;
```

4. To find checkups for a specific physician based on valid test results or admission types with patient details
```
MATCH (p:patient)-[:has]->(c:checkup)<-[:attend]-(ph:physician)
WHERE (c.Admission_Type IN ["Emergency", "Urgent", "Elective"] AND c.Admission_Type = $admissionType)
   OR (c.Test_Results IN ["Abnormal", "Normal", "Inconclusive"] AND c.Test_Results = $testResults)
   AND ph.Physician_Name = $physicianName
RETURN c.Checkup_ID AS CheckupID, 
       c.Diagnosis AS Diagnosis, 
       c.Treatment AS Treatment, 
       c.Test_Results AS TestResults, 
       c.Admission_Type AS AdmissionType, 
       c.Medication AS Medication, 
       p.Name AS PatientName,  
       ph.Physician_Name AS PhysicianName, 
       ph.Specialization AS Specialisation;


Schema:
{schema}

Question:
{question}

Cypher Query:
"""


checkup_cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

checkup_cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    return_intermediate_steps=True,
    verbose=True,
    allow_dangerous_requests=True,
    validate_cypher=True,
    function_response_system="Response based on the Cypher Statment and context",
    cypher_prompt=checkup_cypher_prompt
)