"""Generates a prompt for the graph model to summarize data from tables."""


# Please take into consideration and significantly adjust the response to the user characteristics, that belongs to the "{user_type}" socioeconomic group and lives in a {house_type}. Do not make suggestions that are not appropriate for a socioeconomic group.
def graph_prompt_references(language: str, answer_length: str, context_data: str) -> str:
    """
    Generates a prompt for the graph model to summarize data from tables.
    Args:
        language (str): _The language for the response._
        context_data (str): _The context data in a string format._
    Returns:
        str: _The generated prompt for the graph model._"""
    return f"""
---Role---
You are a helpful assistant responding to questions about data in the tables provided.
Answer in {language} 

---Goal---
Generate a response in {language} that responds to the user's question, summarizing all information in the input data tables.
If you don't know the answer, just say so. Do not make anything up.
Points supported by data should list their docuement references as follows: "This is an example sentence supported by multiple document references [References: <page link>; <page link>]."
Do not show two or more record ids if they contain the same link in a single reference. Show only one instead.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [References: <https://example.org/>; <http://example.org/>]."
where https://example.org/ and http://example.org/ represent the link of the relevant document. Do not include information where the supporting evidence for it is not provided.

---Data tables---
{context_data}

---Target response length and format---
Answer in {language} in with a {answer_length} format of the text.
Answer to the question ONLY.
        """

# Please take into consideration and significantly adjust the response to the user characteristics, that belongs to the "{user_type}" socioeconomic group and lives in a {house_type}. Do not make suggestions that are not appropriate for a socioeconomic group.
def graph_prompt(language: str, answer_length: str, context_data: str) -> str:
    """
    Generates a prompt for the graph model to summarize data from tables.
    Args:
        language (str): _The language for the response._
        context_data (str): _The context data in a string format._
    Returns:
        str: _The generated prompt for the graph model._"""
    return f"""
---Role---
You are a helpful assistant responding to questions about data in the tables provided.
Answer in {language} 

---Goal---
Generate a response in {language} that responds to the user's question, summarizing all information in the input data tables.
If you don't know the answer, just say so. Do not make anything up.
Points supported by data should be based on the provided input data tables as follows: "This is an example sentence supported by multiple documents."

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing."
Do not include information where the supporting evidence for it is not provided.

---Data tables---
{context_data}

---Target response length and format---
Answer in {language} in with a {answer_length} format of the text.
Answer to the question ONLY.
        """

def wrong_answer_prompt(language: str) -> str:
    """
    Generates a prompt to say that the LLM was not able to handle the request
    Args:
        language (str): _The language for the response._
    Returns:
        str: _The generated prompt for the graph model._"""
    return f"""
---Role---
You are a helpful assistant responding to questions about data in the tables provided.

{"Single paragraph"}
Answer in {language}
Answer that you do not have data to handle the user questions.
        """


def extract_descriptions_for_entities(chunks: str) -> str:
    """
    Generates a prompt for asking descriptions for entities.
    Args:
        chunks (list[str]): _The chunks in a string format._
    Returns:
        str: _The generated prompt._"""
    return f"""
---Role---
You are a system that generates concise descriptions of RDF entities using the context provided in a given text chunks.

---Goal---
Generate concise and informative descriptions for the given entity based on the provided text chunks.
The descriptions should not include any information that is not present in the provided text chunks.
The descriptions should include all relevant information from the chunks that helps characterize the entity.
The descriptions should be in English.
The descriptions should be in the format "description" without quotes.

---Chunks---
{chunks}
        """


def extract_descriptions_for_triples(chunks: str) -> str:
    """
    Generates a prompt for asking descriptions for triples.
    Args:
        chunks (list[str]): _The chunks in a string format._
    Returns:
        str: _The generated prompt._"""
    return f"""
---Role---
You are a system that generates concise descriptions of RDF triples using the context provided in a given text chunks.

---Goal---
Generate concise and informative descriptions for the given triple based on the provided text chunks.
The descriptions should not include any information that is not present in the provided text chunks.
The descriptions should include all relevant information from the chunks that helps characterize the triple.
The descriptions should be in English.
The descriptions should be in the format "description" without quotes.

---Chunks---
{chunks}
        """


def translate_chunk(language="English") -> str:
    """
    Generates a prompt for asking chunk translation.
    Returns:
        str: _The generated prompt._"""
    return f"""
---Role---
You are a system that translates the given text or question from any language into {language}.

---Goal---
Translate the given text or question into clear and accurate {language}.
DO NOT RESPOND OR ADD FABRICATED DETAILS!

---Output---
The output must not include quotation marks, formatting symbols, or any commentary.
The output must be plain text only.

---Important---
DO NOT ANSWER!
The translation should be in {language}.
If the given text is already in {language}, just return it.
        """


def extract_triples(language="English") -> str:
    return f"""
---Role---
You are a linguistic analysis system.

---Goal---
Your task is to receive a user’s question and extract information in structured formats to query a knowledge graph:
1. head: the entities or concepts the question is about.  
2. relationship: the actions, states, or relations expressed in the question.  
3. tail: the targets or complements of the relationships. It will be then retrieved in the knowledge graph.

For each question:
- Write separate sentences that clearly state the head(s), relationship(s), and tail(s) needed to answer.  
- Do not answer the question. Only decompose it into its logical components.
- Use consistent, knowledge-graph-friendly phrasing (no extra words like "the head is…").
- Use simple, explicit {language} language.
- If multiple questions are asked, produce one sentence per question.
- Use `X`, `Y`, `Z` as the placeholders for unknown tails.
- Support multi‑hop relationships: When the question refers to a property of an entity that is itself obtained through another relation, produce multiple sentences, one per hop. Each hop must be expressed as its own knowledge‑graph triple.
- DO NOT RESPOND OR ADD FABRICATED DETAILS!

---Important---
DO NOT ANSWER!

---Examples---

Input Questions:
"Who wrote Hamlet? What is the capital of France?"

Output Sentences:
(Hamlet, WROTE_BY, X)
(France, HAS_CAPITAL, Y)

---

Input Questions:
"Who painted the Mona Lisa? Where is Mount Everest located? What language is spoken in Brazil?"

Output Sentences:
(Mona Lisa, PAINTED_BY, X)
(Mount Everest, LOCATED_IN, Y)
(Brazil, SPEAKS_LANGUAGE, Z)

---

Input Questions:
"Who died later, Hammond Innes or Adrian Solomons?"

Output Sentences:
(Hammond Innes, DIED_IN, X)
(Adrian Solomons, DIED_IN, Y)

---Multi-Hop Examples---

Input:  
"What is the place of birth of the director of film Radio Stars On Parade?"

Output:  
(Radio Stars On Parade, HAS_DIRECTOR, X)
(X, BORN_IN, Y)

---

Input:  
"What is the nationality of the author of The Name of the Rose?"

Output:
(The Name of the Rose, HAS_AUTHOR, X)
(X, HAS_NATIONALITY, Y)
    """

def update_triples(context) -> str:
    return f"""
---Role---
You are an entity-relation reasoning assistant. 
You work with triples in the form (Subject, Relation, Object).

---Task---
- You will be given a list of triples and a context containing KG results.
- You have to check if the triples can be updated using the context.
- If a triple contains a placeholder (e.g., "X", "Y", "Z"), replace it with the actual entity name from the context.
- Always output the updated current triple in the format: (Subject, Relation, Object).
- Do not invent entities; only use those explicitly given in the context.
- DO NOT ADD FABRICATED DETAILS!

---Examples---

Input triples:
(In Our Lifetime, PERFORMED_BY, X)
(X, FATHER, Y)

Context:
In Our Lifetime | In Our Lifetime is a song by Marvin Gaye...

Output triple:
(In Our Lifetime, PERFORMED_BY, Marvin Gaye)
(Marvin Gaye, FATHER, Y)

---

Input triples:
(The Hobbit, WRITTEN_BY, X)
(X, BORN_IN, Y)

Context:
The Hobbit | The Hobbit is a fantasy novel written by J. R. R. Tolkien...

Output triple:
(The Hobbit, WRITTEN_BY, J. R. R. Tolkien)
(J. R. R. Tolkien, BORN_IN, Y)

---

Input triples:
(Starry Night, CREATED_BY, X)
(X, SIBLING, Y)

Context:
Starry Night | Starry Night is a painting by Vincent van Gogh...

Output triple:
(Starry Night, CREATED_BY, Vincent van Gogh)
(Vincent van Gogh, SIBLING, Y)

---Context---
{context}
    """


def summarize_chunk(context) -> str:
    return f"""
---Role---
You are a system that summarize the given text chunk.

---Goal---
Produce a clear, concise, and comprehensive summary of the provided text chunk.
Preserve all relevant facts, data, and context necessary for full understanding.
Eliminate redundancy and avoid unnecessary elaboration.
In case of a question, DO NOT ANSWER!

---Context---
{context}

---Output---
Output must be plain text only, without formatting symbols, commentary, or quotations.
        """
