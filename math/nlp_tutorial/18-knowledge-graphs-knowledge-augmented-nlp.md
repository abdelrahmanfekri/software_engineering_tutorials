# NLP Tutorial Module 18: Knowledge Graphs and Knowledge-Augmented NLP

## Learning Objectives
By the end of this module, you will be able to:
- Understand knowledge graph fundamentals and representations
- Build and query knowledge graphs from text
- Implement knowledge-augmented language models
- Apply retrieval-augmented generation (RAG) systems
- Integrate external knowledge bases with neural models
- Build entity linking and knowledge base completion systems

## Introduction to Knowledge Graphs

Knowledge graphs represent structured knowledge as entities and their relationships, enabling machines to reason about facts and their connections.

### Key Concepts

1. **Entities**: Real-world objects, concepts, or events
2. **Relations**: Connections between entities
3. **Triples**: (Subject, Predicate, Object) format
4. **Ontologies**: Formal specifications of knowledge domains
5. **Knowledge Base**: Collection of structured facts

### Knowledge Graph Structure

```
Entity1 --[Relation]--> Entity2
Example: (Apple, headquartered_in, Cupertino)
```

## Building Knowledge Graphs from Text

### Named Entity Recognition and Relation Extraction

```python
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Tuple, Dict, Set

class KnowledgeGraphBuilder:
    def __init__(self, model_name='en_core_web_sm'):
        """Initialize knowledge graph builder with NLP model"""
        self.nlp = spacy.load(model_name)
        self.graph = nx.DiGraph()
        self.entity_types = {}
        self.relation_types = set()
        
    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append((ent.text, ent.label_, ent.start_char, ent.end_char))
            self.entity_types[ent.text] = ent.label_
            
        return entities
    
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relations between entities using dependency parsing"""
        doc = self.nlp(text)
        relations = []
        
        # Simple rule-based relation extraction
        for token in doc:
            if token.pos_ == "VERB":
                subject = None
                obj = None
                
                # Find subject
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        if child.ent_type_:
                            subject = child.text
                    elif child.dep_ in ["dobj", "pobj", "attr"]:
                        if child.ent_type_:
                            obj = child.text
                
                if subject and obj:
                    relation = token.lemma_
                    relations.append((subject, relation, obj))
                    self.relation_types.add(relation)
        
        return relations
    
    def build_from_text(self, texts: List[str]):
        """Build knowledge graph from multiple texts"""
        for text in texts:
            # Extract entities
            entities = self.extract_entities(text)
            
            # Add entities to graph
            for entity, entity_type, _, _ in entities:
                self.graph.add_node(entity, type=entity_type)
            
            # Extract and add relations
            relations = self.extract_relations(text)
            for subject, relation, obj in relations:
                if subject in self.graph.nodes and obj in self.graph.nodes:
                    self.graph.add_edge(subject, obj, relation=relation)
    
    def query_graph(self, entity: str, max_depth: int = 2) -> Dict:
        """Query knowledge graph for entity and its connections"""
        if entity not in self.graph:
            return {"error": "Entity not found"}
        
        subgraph = nx.ego_graph(self.graph, entity, radius=max_depth)
        
        results = {
            "entity": entity,
            "type": self.entity_types.get(entity, "Unknown"),
            "neighbors": [],
            "relations": []
        }
        
        for neighbor in subgraph.neighbors(entity):
            edge_data = self.graph[entity][neighbor]
            results["neighbors"].append(neighbor)
            results["relations"].append({
                "target": neighbor,
                "relation": edge_data.get("relation", "unknown")
            })
        
        return results
    
    def visualize(self, max_nodes: int = 50, figsize: Tuple[int, int] = (15, 10)):
        """Visualize knowledge graph"""
        # Limit graph size for visualization
        if len(self.graph.nodes) > max_nodes:
            # Take most connected nodes
            degrees = dict(self.graph.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            subgraph = self.graph.subgraph(top_nodes)
        else:
            subgraph = self.graph
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Draw nodes with colors based on entity type
        entity_type_colors = {
            'PERSON': 'lightblue',
            'ORG': 'lightgreen',
            'GPE': 'lightcoral',
            'DATE': 'lightyellow',
            'MONEY': 'lightpink'
        }
        
        node_colors = [entity_type_colors.get(self.entity_types.get(node, ''), 'lightgray') 
                      for node in subgraph.nodes()]
        
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                              node_size=1000, alpha=0.7)
        nx.draw_networkx_labels(subgraph, pos, font_size=8)
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.5)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(subgraph, 'relation')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=6)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def export_triples(self) -> List[Tuple[str, str, str]]:
        """Export knowledge graph as triples"""
        triples = []
        for source, target, data in self.graph.edges(data=True):
            relation = data.get('relation', 'related_to')
            triples.append((source, relation, target))
        return triples
    
    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relations": self.graph.number_of_edges(),
            "entity_types": dict(Counter(self.entity_types.values())),
            "relation_types": list(self.relation_types),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "density": nx.density(self.graph)
        }

# Example usage
from collections import Counter

texts = [
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Steve Jobs served as CEO of Apple from 1997 to 2011.",
    "Apple is headquartered in Cupertino and manufactures iPhones.",
    "Tim Cook became CEO of Apple after Steve Jobs.",
    "Microsoft was founded by Bill Gates in Seattle.",
    "Bill Gates served as CEO of Microsoft for many years."
]

# Build knowledge graph
kg_builder = KnowledgeGraphBuilder()
kg_builder.build_from_text(texts)

# Get statistics
stats = kg_builder.get_statistics()
print("Knowledge Graph Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Query the graph
query_result = kg_builder.query_graph("Apple")
print(f"\nQuery results for 'Apple':")
print(query_result)

# Export triples
triples = kg_builder.export_triples()
print(f"\nKnowledge triples (first 10):")
for triple in triples[:10]:
    print(f"  {triple}")

# Visualize
kg_builder.visualize()
```

## Knowledge Graph Embeddings

### TransE: Translational Embedding Model

```python
class TransE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 100):
        super(TransE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Normalize embeddings
        self.entity_embeddings.weight.data = F.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )
    
    def forward(self, heads, relations, tails):
        """
        Forward pass for TransE model
        h + r ≈ t
        """
        head_embeds = self.entity_embeddings(heads)
        relation_embeds = self.relation_embeddings(relations)
        tail_embeds = self.entity_embeddings(tails)
        
        # TransE scoring: ||h + r - t||
        scores = head_embeds + relation_embeds - tail_embeds
        scores = torch.norm(scores, p=2, dim=1)
        
        return scores
    
    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """Get embedding for specific entity"""
        return self.entity_embeddings(torch.tensor([entity_id]))
    
    def get_relation_embedding(self, relation_id: int) -> torch.Tensor:
        """Get embedding for specific relation"""
        return self.relation_embeddings(torch.tensor([relation_id]))
    
    def predict_tail(self, head: int, relation: int, top_k: int = 10):
        """Predict most likely tail entities for (head, relation, ?)"""
        head_embed = self.entity_embeddings(torch.tensor([head]))
        relation_embed = self.relation_embeddings(torch.tensor([relation]))
        
        # Calculate scores for all possible tails
        all_tail_embeds = self.entity_embeddings.weight
        predicted = head_embed + relation_embed
        
        # Compute distances to all entities
        distances = torch.norm(
            predicted.unsqueeze(1) - all_tail_embeds.unsqueeze(0), 
            p=2, dim=2
        )
        
        # Get top-k closest entities
        top_k_distances, top_k_indices = torch.topk(
            -distances[0], k=top_k
        )
        
        return top_k_indices.tolist(), -top_k_distances.tolist()

class KnowledgeGraphEmbeddingTrainer:
    def __init__(self, triples: List[Tuple[str, str, str]], embedding_dim: int = 100):
        """Initialize trainer with knowledge graph triples"""
        self.triples = triples
        self.embedding_dim = embedding_dim
        
        # Create entity and relation vocabularies
        entities = set()
        relations = set()
        
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        
        self.entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}
        self.relation_to_id = {r: i for i, r in enumerate(sorted(relations))}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}
        
        # Convert triples to IDs
        self.triple_ids = [
            (self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t])
            for h, r, t in triples
        ]
        
        # Initialize model
        self.model = TransE(
            num_entities=len(self.entity_to_id),
            num_relations=len(self.relation_to_id),
            embedding_dim=embedding_dim
        )
    
    def generate_negative_samples(self, positive_triples: List[Tuple[int, int, int]], 
                                 num_negative: int = 1):
        """Generate negative samples by corrupting positive triples"""
        negative_triples = []
        
        for h, r, t in positive_triples:
            for _ in range(num_negative):
                # Randomly corrupt head or tail
                if np.random.rand() < 0.5:
                    # Corrupt head
                    corrupted_h = np.random.randint(0, self.model.num_entities)
                    negative_triples.append((corrupted_h, r, t))
                else:
                    # Corrupt tail
                    corrupted_t = np.random.randint(0, self.model.num_entities)
                    negative_triples.append((h, r, corrupted_t))
        
        return negative_triples
    
    def train(self, epochs: int = 100, batch_size: int = 128, 
             learning_rate: float = 0.01, margin: float = 1.0):
        """Train TransE model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(self.triple_ids)
            
            for i in range(0, len(self.triple_ids), batch_size):
                batch_triples = self.triple_ids[i:i+batch_size]
                
                # Generate negative samples
                negative_triples = self.generate_negative_samples(batch_triples)
                
                # Convert to tensors
                pos_heads = torch.tensor([h for h, r, t in batch_triples])
                pos_rels = torch.tensor([r for h, r, t in batch_triples])
                pos_tails = torch.tensor([t for h, r, t in batch_triples])
                
                neg_heads = torch.tensor([h for h, r, t in negative_triples])
                neg_rels = torch.tensor([r for h, r, t in negative_triples])
                neg_tails = torch.tensor([t for h, r, t in negative_triples])
                
                # Forward pass
                pos_scores = self.model(pos_heads, pos_rels, pos_tails)
                neg_scores = self.model(neg_heads, neg_rels, neg_tails)
                
                # Margin ranking loss
                loss = torch.mean(F.relu(margin + pos_scores - neg_scores))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Normalize entity embeddings
                self.model.entity_embeddings.weight.data = F.normalize(
                    self.model.entity_embeddings.weight.data, p=2, dim=1
                )
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(self.triple_ids) / batch_size)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def visualize_embeddings(self, entities: List[str] = None, method: str = 'tsne'):
        """Visualize entity embeddings in 2D"""
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Get all entity embeddings
        all_embeddings = self.model.entity_embeddings.weight.detach().numpy()
        
        # Select entities to visualize
        if entities is None:
            entities = list(self.entity_to_id.keys())[:50]
        
        entity_ids = [self.entity_to_id[e] for e in entities if e in self.entity_to_id]
        embeddings = all_embeddings[entity_ids]
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
        
        for i, entity in enumerate([e for e in entities if e in self.entity_to_id]):
            plt.annotate(entity, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.title(f'Knowledge Graph Entity Embeddings ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.show()

# Train knowledge graph embeddings
kg_embeddings = KnowledgeGraphEmbeddingTrainer(triples, embedding_dim=50)
losses = kg_embeddings.train(epochs=100, batch_size=32, learning_rate=0.01)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('TransE Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Visualize embeddings
kg_embeddings.visualize_embeddings()

# Make predictions
apple_id = kg_embeddings.entity_to_id.get('Apple', 0)
founded_by_id = kg_embeddings.relation_to_id.get('found', 0)

predicted_entities, scores = kg_embeddings.model.predict_tail(apple_id, founded_by_id, top_k=5)
print("\nPredicted entities for (Apple, founded_by, ?):")
for entity_id, score in zip(predicted_entities, scores):
    entity_name = kg_embeddings.id_to_entity[entity_id]
    print(f"  {entity_name}: {score:.4f}")
```

## Retrieval-Augmented Generation (RAG)

### RAG Architecture Implementation

```python
import faiss
from sentence_transformers import SentenceTransformer

class RetrievalAugmentedGenerator:
    def __init__(self, knowledge_base: List[str], 
                 retriever_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 generator_model: str = 'gpt2'):
        """
        Initialize RAG system
        
        Args:
            knowledge_base: List of documents to retrieve from
            retriever_model: Model for encoding queries and documents
            generator_model: Model for text generation
        """
        self.knowledge_base = knowledge_base
        
        # Initialize retriever
        self.retriever = SentenceTransformer(retriever_model)
        
        # Encode knowledge base
        print("Encoding knowledge base...")
        self.kb_embeddings = self.retriever.encode(knowledge_base, show_progress_bar=True)
        
        # Build FAISS index
        self.dimension = self.kb_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.kb_embeddings.astype('float32'))
        
        # Initialize generator
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = AutoModelForCausalLM.from_pretrained(generator_model)
        
        print(f"RAG system initialized with {len(knowledge_base)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve most relevant documents for query"""
        # Encode query
        query_embedding = self.retriever.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return documents and scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((self.knowledge_base[idx], float(distance)))
        
        return results
    
    def generate(self, query: str, retrieved_docs: List[str], 
                max_length: int = 100) -> str:
        """Generate answer using query and retrieved documents"""
        # Construct prompt with retrieved context
        context = "\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
        prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate
        outputs = self.generator.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (text after "Answer:")
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        return answer
    
    def answer_question(self, question: str, top_k: int = 3, 
                       max_length: int = 100) -> Dict:
        """Complete RAG pipeline: retrieve + generate"""
        # Retrieve relevant documents
        retrieved_docs_with_scores = self.retrieve(question, top_k=top_k)
        retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
        
        # Generate answer
        answer = self.generate(question, retrieved_docs, max_length=max_length)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": [
                {"document": doc, "score": score} 
                for doc, score in retrieved_docs_with_scores
            ]
        }

# Example knowledge base
knowledge_base = [
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
    "Apple is headquartered in Cupertino, California.",
    "The iPhone was first released by Apple in 2007.",
    "Steve Jobs served as CEO of Apple from 1997 until 2011.",
    "Tim Cook became CEO of Apple in 2011 after Steve Jobs.",
    "Apple manufactures various products including Mac computers, iPhones, iPads, and Apple Watches.",
    "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
    "Microsoft is headquartered in Redmond, Washington.",
    "Bill Gates served as CEO of Microsoft until 2000.",
    "Satya Nadella became CEO of Microsoft in 2014."
]

# Initialize RAG system
rag_system = RetrievalAugmentedGenerator(knowledge_base)

# Answer questions
questions = [
    "Who founded Apple?",
    "When did Tim Cook become CEO?",
    "What products does Apple make?"
]

for question in questions:
    result = rag_system.answer_question(question, top_k=3)
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print("Retrieved documents:")
    for i, doc_info in enumerate(result['retrieved_documents'], 1):
        print(f"  {i}. {doc_info['document']} (score: {doc_info['score']:.4f})")
```

## Knowledge-Augmented Language Models

### BERT with Knowledge Graph Integration

```python
class KnowledgeAugmentedBERT(nn.Module):
    def __init__(self, bert_model: str = 'bert-base-uncased',
                 kg_embedding_dim: int = 100,
                 num_entities: int = 1000,
                 fusion_method: str = 'concat'):
        super(KnowledgeAugmentedBERT, self).__init__()
        
        # Load BERT
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert_dim = self.bert.config.hidden_size
        
        # Knowledge graph embeddings
        self.entity_embeddings = nn.Embedding(num_entities, kg_embedding_dim)
        
        # Fusion layer
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            self.fusion = nn.Linear(self.bert_dim + kg_embedding_dim, self.bert_dim)
        elif fusion_method == 'attention':
            self.kg_attention = nn.MultiheadAttention(self.bert_dim, num_heads=8)
            self.kg_projection = nn.Linear(kg_embedding_dim, self.bert_dim)
        
        # Output layer
        self.classifier = nn.Linear(self.bert_dim, 2)  # Binary classification
        
    def forward(self, input_ids, attention_mask, entity_ids=None):
        """
        Forward pass with knowledge augmentation
        
        Args:
            input_ids: BERT input token IDs
            attention_mask: BERT attention mask
            entity_ids: Entity IDs from knowledge graph (optional)
        """
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_representation = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        if entity_ids is not None and len(entity_ids) > 0:
            # Get entity embeddings
            entity_embeds = self.entity_embeddings(entity_ids)
            
            # Fusion
            if self.fusion_method == 'concat':
                # Concatenate text and knowledge
                if entity_embeds.dim() == 3:
                    entity_embeds = entity_embeds.mean(dim=1)  # Average entity embeddings
                combined = torch.cat([text_representation, entity_embeds], dim=1)
                fused_representation = self.fusion(combined)
            elif self.fusion_method == 'attention':
                # Attention-based fusion
                entity_embeds_projected = self.kg_projection(entity_embeds)
                if entity_embeds_projected.dim() == 2:
                    entity_embeds_projected = entity_embeds_projected.unsqueeze(1)
                
                text_query = text_representation.unsqueeze(0)
                attended, _ = self.kg_attention(
                    text_query, 
                    entity_embeds_projected.transpose(0, 1),
                    entity_embeds_projected.transpose(0, 1)
                )
                fused_representation = attended.squeeze(0)
        else:
            fused_representation = text_representation
        
        # Classification
        logits = self.classifier(fused_representation)
        
        return logits

# Example usage
model = KnowledgeAugmentedBERT(
    bert_model='bert-base-uncased',
    kg_embedding_dim=100,
    num_entities=1000,
    fusion_method='attention'
)

# Sample input
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text = "Apple Inc. was founded by Steve Jobs."
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Entity IDs (e.g., from entity linking)
entity_ids = torch.tensor([[1, 5, 10]])  # IDs for Apple, Steve Jobs, etc.

# Forward pass
logits = model(inputs['input_ids'], inputs['attention_mask'], entity_ids)
print(f"Output logits: {logits}")
```

## Entity Linking and Disambiguation

```python
class EntityLinker:
    def __init__(self, entity_catalog: Dict[str, Dict], 
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize entity linker
        
        Args:
            entity_catalog: Dictionary mapping entity mentions to entity information
                           Format: {entity_id: {'name': str, 'aliases': List[str], 'description': str}}
            embedding_model: Model for encoding entity descriptions
        """
        self.entity_catalog = entity_catalog
        self.encoder = SentenceTransformer(embedding_model)
        
        # Build entity index
        self.entity_ids = list(entity_catalog.keys())
        self.entity_descriptions = [
            f"{info['name']} {info.get('description', '')}"
            for info in entity_catalog.values()
        ]
        
        # Encode entity descriptions
        self.entity_embeddings = self.encoder.encode(self.entity_descriptions)
        
        # Build FAISS index
        self.dimension = self.entity_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.entity_embeddings.astype('float32'))
        
        # Build mention to entity mapping
        self.mention_to_entities = defaultdict(set)
        for entity_id, info in entity_catalog.items():
            self.mention_to_entities[info['name'].lower()].add(entity_id)
            for alias in info.get('aliases', []):
                self.mention_to_entities[alias.lower()].add(entity_id)
    
    def link_entities(self, text: str, mentions: List[Tuple[str, int, int]] = None) -> List[Dict]:
        """
        Link entity mentions in text to knowledge base entities
        
        Args:
            text: Input text
            mentions: List of (mention_text, start, end) tuples. If None, extract automatically.
        
        Returns:
            List of linked entities with confidence scores
        """
        # Extract mentions if not provided
        if mentions is None:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            mentions = [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents]
        
        linked_entities = []
        
        for mention_text, start, end in mentions:
            # Get context around mention
            context_start = max(0, start - 100)
            context_end = min(len(text), end + 100)
            context = text[context_start:context_end]
            
            # Candidate generation
            candidates = self.mention_to_entities.get(mention_text.lower(), set())
            
            if not candidates:
                # Use semantic search for candidates
                query_embedding = self.encoder.encode([f"{mention_text} {context}"])
                distances, indices = self.index.search(query_embedding.astype('float32'), k=5)
                candidates = {self.entity_ids[idx] for idx in indices[0]}
            
            # Disambiguation using context
            best_entity = None
            best_score = float('inf')
            
            for entity_id in candidates:
                entity_desc = self.entity_descriptions[self.entity_ids.index(entity_id)]
                
                # Compute similarity between context and entity description
                combined_query = f"{mention_text} {context}"
                query_emb = self.encoder.encode([combined_query])
                entity_emb = self.entity_embeddings[self.entity_ids.index(entity_id)].reshape(1, -1)
                
                # Compute distance
                distance = np.linalg.norm(query_emb - entity_emb)
                
                if distance < best_score:
                    best_score = distance
                    best_entity = entity_id
            
            linked_entities.append({
                'mention': mention_text,
                'start': start,
                'end': end,
                'entity_id': best_entity,
                'entity_name': self.entity_catalog[best_entity]['name'] if best_entity else None,
                'confidence': 1.0 / (1.0 + best_score)  # Convert distance to confidence
            })
        
        return linked_entities

# Example entity catalog
entity_catalog = {
    'E1': {
        'name': 'Apple Inc.',
        'aliases': ['Apple', 'AAPL'],
        'description': 'American technology company that manufactures consumer electronics and software'
    },
    'E2': {
        'name': 'Steve Jobs',
        'aliases': ['Jobs'],
        'description': 'Co-founder and former CEO of Apple Inc.'
    },
    'E3': {
        'name': 'Apple (fruit)',
        'aliases': ['apple'],
        'description': 'Edible fruit produced by apple trees'
    },
    'E4': {
        'name': 'Tim Cook',
        'aliases': ['Cook'],
        'description': 'Current CEO of Apple Inc.'
    }
}

# Initialize entity linker
entity_linker = EntityLinker(entity_catalog)

# Link entities in text
test_text = "Apple Inc. was founded by Steve Jobs. Tim Cook is the current CEO of Apple."
linked_entities = entity_linker.link_entities(test_text)

print("Linked Entities:")
for entity in linked_entities:
    print(f"  Mention: '{entity['mention']}' -> Entity: {entity['entity_name']} "
          f"(ID: {entity['entity_id']}, Confidence: {entity['confidence']:.4f})")
```

## Knowledge Base Completion

### Neural Link Prediction

```python
class KnowledgeBaseCompletion(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, 
                 embedding_dim: int = 200, hidden_dim: int = 400):
        super(KnowledgeBaseCompletion, self).__init__()
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(self, heads, relations, tails):
        """Predict probability of triple being true"""
        head_embeds = self.entity_embeddings(heads)
        relation_embeds = self.relation_embeddings(relations)
        tail_embeds = self.entity_embeddings(tails)
        
        # Concatenate embeddings
        combined = torch.cat([head_embeds, relation_embeds, tail_embeds], dim=1)
        
        # Predict probability
        scores = self.predictor(combined)
        
        return scores.squeeze()
    
    def predict_missing_links(self, head: int, relation: int, 
                             num_entities: int, top_k: int = 10):
        """Predict most likely tail entities for (head, relation, ?)"""
        # Prepare batch input
        heads = torch.tensor([head] * num_entities)
        relations = torch.tensor([relation] * num_entities)
        tails = torch.arange(num_entities)
        
        # Get predictions
        with torch.no_grad():
            scores = self.forward(heads, relations, tails)
        
        # Get top-k predictions
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        return top_indices.tolist(), top_scores.tolist()

# Train knowledge base completion model
def train_kb_completion(model, train_triples, num_entities, 
                       epochs=100, batch_size=128, learning_rate=0.001):
    """Train knowledge base completion model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(train_triples)
        
        for i in range(0, len(train_triples), batch_size):
            batch = train_triples[i:i+batch_size]
            
            # Positive samples
            pos_heads = torch.tensor([h for h, r, t in batch])
            pos_rels = torch.tensor([r for h, r, t in batch])
            pos_tails = torch.tensor([t for h, r, t in batch])
            pos_labels = torch.ones(len(batch))
            
            # Negative samples (corrupt tails)
            neg_tails = torch.randint(0, num_entities, (len(batch),))
            neg_labels = torch.zeros(len(batch))
            
            # Combine positive and negative
            all_heads = torch.cat([pos_heads, pos_heads])
            all_rels = torch.cat([pos_rels, pos_rels])
            all_tails = torch.cat([pos_tails, neg_tails])
            all_labels = torch.cat([pos_labels, neg_labels])
            
            # Forward pass
            scores = model(all_heads, all_rels, all_tails)
            loss = criterion(scores, all_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / (len(train_triples) / batch_size)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# Example: Complete knowledge base
kb_completion_model = KnowledgeBaseCompletion(
    num_entities=len(kg_embeddings.entity_to_id),
    num_relations=len(kg_embeddings.relation_to_id),
    embedding_dim=100
)

# Train model
train_kb_completion(kb_completion_model, kg_embeddings.triple_ids, 
                   len(kg_embeddings.entity_to_id), epochs=50)

# Predict missing links
apple_id = kg_embeddings.entity_to_id.get('Apple', 0)
ceo_relation_id = kg_embeddings.relation_to_id.get('ceo', 0)

predicted_entities, scores = kb_completion_model.predict_missing_links(
    apple_id, ceo_relation_id, len(kg_embeddings.entity_to_id), top_k=5
)

print("\nPredicted entities for (Apple, CEO, ?):")
for entity_id, score in zip(predicted_entities, scores):
    if entity_id in kg_embeddings.id_to_entity:
        entity_name = kg_embeddings.id_to_entity[entity_id]
        print(f"  {entity_name}: {score:.4f}")
```

## Practical Exercises

### Exercise 1: Knowledge Graph Construction
Build a knowledge graph from a large text corpus:
- Extract entities and relations from news articles
- Handle coreference resolution
- Implement temporal information extraction
- Create a queryable knowledge base

### Exercise 2: Knowledge-Enhanced QA System
Implement a question-answering system that uses knowledge graphs:
- Build entity linking for questions
- Retrieve relevant subgraphs
- Generate answers using both text and structured knowledge
- Evaluate against baseline models

### Exercise 3: Knowledge Graph Completion
Implement and compare multiple knowledge graph embedding methods:
- TransE, TransR, DistMult, ComplEx
- Evaluate on link prediction task
- Analyze which methods work best for different relation types

## Assessment Questions

1. **What are the advantages of knowledge graphs over unstructured text?**
   - Structured representation of facts
   - Enable logical reasoning
   - Support complex queries
   - Facilitate knowledge integration

2. **How does RAG improve language model performance?**
   - Provides external knowledge
   - Reduces hallucinations
   - Enables knowledge updates without retraining
   - Improves factual accuracy

3. **What are the challenges in entity linking?**
   - Name ambiguity
   - Mention variations
   - Context dependency
   - Coverage of knowledge base

## Key Takeaways

- Knowledge graphs provide structured representations of world knowledge
- Knowledge graph embeddings enable reasoning over symbolic knowledge
- RAG systems combine retrieval and generation for better factual accuracy
- Knowledge-augmented models outperform pure neural approaches on knowledge-intensive tasks
- Entity linking bridges unstructured text and structured knowledge
- Knowledge base completion can discover new facts from existing knowledge

## Next Steps

In the next module, we'll explore Agentic AI systems that use knowledge graphs and reasoning to plan and execute complex tasks autonomously.

