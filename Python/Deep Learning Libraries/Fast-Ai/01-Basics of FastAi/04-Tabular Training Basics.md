
We will use the example of the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/Adult) where we have to predict if a person is earning more or less than $50k per year using some general data.

```python
from fastai.tabular.all import *
import pandas as pd

path=untar_data(URLs.ADULT_SAMPLES)
path.ls()

df=pd.read_csv(path/"adult.csv")
df.head()
```

|     | age | workclass        | fnlwgt | education   | education-num | marital-status     | occupation      | relationship  | race               | sex    | capital-gain | capital-loss | hours-per-week | native-country | salary |
| --- | --- | ---------------- | ------ | ----------- | ------------- | ------------------ | --------------- | ------------- | ------------------ | ------ | ------------ | ------------ | -------------- | -------------- | ------ |
| 0   | 49  | Private          | 101320 | Assoc-acdm  | 12.0          | Married-civ-spouse | NaN             | Wife          | White              | Female | 0            | 1902         | 40             | United-States  | >=50k  |
| 1   | 44  | Private          | 236746 | Masters     | 14.0          | Divorced           | Exec-managerial | Not-in-family | White              | Male   | 10520        | 0            | 45             | United-States  | >=50k  |
| 2   | 38  | Private          | 96185  | HS-grad     | NaN           | Divorced           | NaN             | Unmarried     | Black              | Female | 0            | 0            | 32             | United-States  | <50k   |
| 3   | 38  | Self-emp-inc     | 112847 | Prof-school | 15.0          | Married-civ-spouse | Prof-specialty  | Husband       | Asian-Pac-Islander | Male   | 0            | 0            | 40             | United-States  | >=50k  |
| 4   | 42  | Self-emp-not-inc | 82297  | 7th-8th     | NaN           | Married-civ-spouse | Other-service   | Wife          | Black              | Female | 0            | 0            | 50             | United-States  | <50k   |

```python
# Creating DataLoaders
dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path,
								  y_names="salary",
								  cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
								  cont_names = ['age', 'fnlwgt', 
								  'education-num'],
								  procs = [Categorify, FillMissing, 
								  Normalize])
```

The last part is the list of pre-processors we apply to our data:

- [`Categorify`](https://docs.fast.ai/tabular.core.html#categorify) is going to take every categorical variable and make a map from integer to unique categories, then replace the values by the corresponding index.
- [`FillMissing`](https://docs.fast.ai/tabular.core.html#fillmissing) will fill the missing values in the continuous variables by the median of existing values (you can choose a specific value if you prefer)
- [`Normalize`](https://docs.fast.ai/data.transforms.html#normalize) will normalize the continuous variables (subtract the mean and divide by the std)

To further expose what’s going on below the surface, let’s rewrite this utilizing `fastai`’s [`TabularPandas`](https://docs.fast.ai/tabular.core.html#tabularpandas) class. We will need to make one adjustment, which is defining how we want to split our data. By default the factory method above used a random 80/20 split, so we will do the same:

```python
splits = RandomSplitter(valid_pct=0.2)(range_of(df))

to = TabularPandas(df, procs=[Categorify, FillMissing,Normalize],
                   cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
                   cont_names = ['age', 'fnlwgt', 'education-num'],
                   y_names='salary',splits=splits)

to.xs.iloc[:2]
```

We can define a model using the [`tabular_learner`](https://docs.fast.ai/tabular.learner.html#tabular_learner) method. When we define our model, `fastai` will try to infer the loss function based on our `y_names` earlier.

**Note**: Sometimes with tabular data, your `y`’s may be encoded (such as 0 and 1). In such a case you should explicitly pass `y_block = CategoryBlock` in your constructor so `fastai` won’t presume you are doing regression.

#### Continue the training:

```python
dls.show_batch()

learner=tabular_learner(dls,metrics=accuracy)
learner.fit_one_cycle(5,1e-2)

learner.show_results()

row, clas, probs = learn.predict(df.iloc[0])

```

