import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import ifcopenshell
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

def parse_ifc_files(filename):
    try:
        ifc_file = ifcopenshell.open(filename)
        entities = ifc_file.by_type('IfcEntity')
        entity_names = [entity.Name if hasattr(entity, 'Name') and entity.Name else 'Unnamed' for entity in entities]
        attributes = [entity.get_info() for entity in entities]
        return entity_names, attributes
    except Exception as e:
        logger.error(f"Error parsing IFC file {filename}: {e}")
        return [], []

def create_adjacency_list(entity_names):
    num_entities = len(entity_names)
    # List of tuples representing edges between nodes (assuming every node is connected to every other node)
    adjacency_list = [(i, j) for i in range(num_entities) for j in range(num_entities) if i != j]
    return adjacency_list

def create_node_features(attributes):
    node_features = [len(attr) for attr in attributes]
    return np.array(node_features)

def tokenize_and_embed(node_features):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(node_features, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
    return criterion(preds, data.y).item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred == data.y
    test_acc = accuracy_score(data.y, pred)
    precision = precision_score(data.y, pred, average='weighted')
    recall = recall_score(data.y, pred, average='weighted')
    f1 = f1_score(data.y, pred, average='weighted')
    return test_acc, precision, recall, f1

def main():
    ifc_files = ["ifc_train_file.ifc", "ifc_val_file.ifc", "ifc_test_file.ifc"]
    data_list = []
    for file in ifc_files:
        if not os.path.isfile(file):
            logger.error(f"IFC file {file} does not exist.")
            continue
        entity_names, attributes = parse_ifc_files(file)
        adj_list, node_features = create_adjacency_list(entity_names), create_node_features(attributes)
        embeddings = tokenize_and_embed(node_features)
        data = Data(x=torch.tensor(embeddings, dtype=torch.float), edge_index=torch.tensor(adj_list, dtype=torch.long))
        data_list.append(data)

    # Verify data loaded correctly
    if len(data_list) != 3:
        logger.error(f"Error loading IFC files. Expected 3 files, found {len(data_list)}")
        return

    train_data, val_data, test_data = data_list

    model = GNN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        loss = train(model, train_data, optimizer, criterion)
        logger.info(f"Epoch {epoch+1}: Loss = {loss}")

        if epoch % 10 == 0:  # Save model every 10 epochs
            torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')

    val_loss = validate(model, val_data, criterion)
    logger.info(f'Validation Loss: {val_loss}')
    test_acc, precision, recall, f1 = test(model, test_data)
    logger.info(f'Test Accuracy: {test_acc}, Precision: {precision}, Recall: {recall}, F1-score: {f1}')

    torch.save(model.state_dict(), 'final_model.pth')

if __name__ == "__main__":
    main()
