# %% 

from dotenv import load_dotenv
import os

# %%
load_dotenv()
GOOGLE_AI_KEY = os.getenv("GOOGLE_API_KEY")
print(GOOGLE_AI_KEY) 
# %%
