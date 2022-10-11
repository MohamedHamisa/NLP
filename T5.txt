!pip install simplet5

# let's get a dataset
import pandas as pd
from sklearn.model_selection import train_test_split

path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
df = pd.read_csv(path)
df.head()

# simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
df = df.rename(columns={"headlines":"target_text", "text":"source_text"})
df = df[['source_text', 'target_text']]

# T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
df['source_text'] = "summarize: " + df['source_text']
df

train_df, test_df = train_test_split(df, test_size=0.2)
train_df.shape, test_df.shape

from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
model.train(train_df=train_df[:5000],
            eval_df=test_df[:100], 
            source_max_token_len=128, 
            target_max_token_len=50, 
            batch_size=8, max_epochs=3, use_gpu=True)

# let's load the trained model for inferencing:
model.load_model("t5","outputs/SimpleT5-epoch-2-train-loss-0.9478", use_gpu=True)

text_to_summarize="""summarize: Jo Gandhi has replied to Goa CM Manohar Parrikar's letter, 
which accused the Congress President of using his "visit to an ailing man for political gains". 
"He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me," 
Kevin wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Jo.
"""
model.predict(text_to_summarize)
