# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Build a Named Entity Recognition (NER) model that can automatically identify and classify entities like names of people, locations, organizations, and other important terms from text. The goal is to tag each word in a sentence with its corresponding entity label.

<img width="247" height="654" alt="image" src="https://github.com/user-attachments/assets/2dc00e38-46b0-4be3-86e0-0369d10d6212" />


## DESIGN STEPS
1.Data Preparation: Load the NER dataset, group words into sentences, convert them to numerical indices, and apply padding to make all sequences equal length.

2.Model Construction: Build a Bidirectional LSTM (BiLSTM) neural network with an embedding layer and a fully connected layer to perform sequence labeling for named entities.

3.Training and Evaluation: Train the model using CrossEntropyLoss and Adam optimizer, evaluate its performance on the test dataset, and generate predictions for entity classification.


## PROGRAM
### Name:  KUKKADAPU CHARAN TEJ
### Register Number: 212224040167
```python
class BiLSTMTagger(nn.Module):

    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        lstm_out, _ = self.lstm(x)

        outputs = self.fc(lstm_out)

        return outputs

model = BiLSTMTagger(vocab_size, num_tags).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            loss = loss_fn(
                outputs.view(-1, outputs.shape[-1]),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in test_loader:

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                loss = loss_fn(
                    outputs.view(-1, outputs.shape[-1]),
                    labels.view(-1)
                )

                val_loss += loss.item()

        val_loss = val_loss / len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print("Train Loss:", train_loss)
        print("Val Loss:", val_loss)

    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="549" height="89" alt="image" src="https://github.com/user-attachments/assets/880e0eb4-55c5-4092-bc27-29dddd85734c" />


<img width="935" height="630" alt="image" src="https://github.com/user-attachments/assets/add40235-eee8-47f5-8b3a-e1fbe481dc58" />


### Sample Text Prediction

<img width="416" height="689" alt="image" src="https://github.com/user-attachments/assets/4357d087-29bb-4f23-ab80-5922dbff7d1f" />




## RESULT

The BiLSTM NER model achieved good accuracy in identifying entities like persons, locations, and organizations. It showed strong performance on frequent tags, with scope for improvement on rarer ones.

