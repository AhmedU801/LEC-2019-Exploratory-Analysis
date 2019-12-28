# Exploratory Analysis

As a fan I wanted to have a little fun visulizing and exploring this LEC data set. If you see any mistakes let me know or maybe there is a question that I could have asked. 

credit: https://www.kaggle.com/stephenofarrell/league-of-legends-european-championship-2019


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
```


```python
raw_champion_data = pd.read_csv('lec_championdata.csv')
raw_match_data = pd.read_csv('lec_matchdata.csv')
raw_player_data = pd.read_csv('lec_playerdata.csv')
```

# Preprocessing

## Champion Data


```python
champion_data = raw_champion_data.copy().drop(['Unnamed: 0', ' 0'], axis=1)
champion_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Champion 1</th>
      <th>∑ 2</th>
      <th>W 3</th>
      <th>L 4</th>
      <th>WR 5</th>
      <th>∑ 6</th>
      <th>W 7</th>
      <th>L 8</th>
      <th>WR 9</th>
      <th>∑ 10</th>
      <th>W 11</th>
      <th>L 12</th>
      <th>WR 13</th>
      <th>&lt;25 games</th>
      <th>&lt;25 winrate</th>
      <th>&lt;25 win percentage</th>
      <th>25-30 games</th>
      <th>25-30 winrate</th>
      <th>25-30 win percentage</th>
      <th>30-35 games</th>
      <th>30-35 winrate</th>
      <th>30-35 win percentage</th>
      <th>35-40 games</th>
      <th>35-40 winrate</th>
      <th>35-40 win percentage</th>
      <th>40-45 games</th>
      <th>40-45 winrate</th>
      <th>40-45 win percentage</th>
      <th>&gt;45 games</th>
      <th>&gt;45 winrate</th>
      <th>&gt;45 win percentage</th>
      <th>Split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jarvan IV</td>
      <td>44</td>
      <td>22</td>
      <td>22</td>
      <td>50</td>
      <td>15</td>
      <td>8</td>
      <td>7</td>
      <td>53</td>
      <td>29</td>
      <td>14</td>
      <td>15</td>
      <td>48</td>
      <td>2</td>
      <td>1-1</td>
      <td>50</td>
      <td>16</td>
      <td>9-7</td>
      <td>56</td>
      <td>10</td>
      <td>4-6</td>
      <td>40</td>
      <td>5</td>
      <td>3-2</td>
      <td>60</td>
      <td>9</td>
      <td>5-4</td>
      <td>56</td>
      <td>2</td>
      <td>0-2</td>
      <td>0</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Braum</td>
      <td>44</td>
      <td>19</td>
      <td>25</td>
      <td>43</td>
      <td>23</td>
      <td>10</td>
      <td>13</td>
      <td>43</td>
      <td>21</td>
      <td>9</td>
      <td>12</td>
      <td>43</td>
      <td>4</td>
      <td>4-0</td>
      <td>100</td>
      <td>12</td>
      <td>3-9</td>
      <td>25</td>
      <td>16</td>
      <td>7-9</td>
      <td>44</td>
      <td>4</td>
      <td>0-4</td>
      <td>0</td>
      <td>7</td>
      <td>4-3</td>
      <td>57</td>
      <td>1</td>
      <td>1-0</td>
      <td>100</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alistar</td>
      <td>39</td>
      <td>21</td>
      <td>18</td>
      <td>54</td>
      <td>19</td>
      <td>9</td>
      <td>10</td>
      <td>47</td>
      <td>20</td>
      <td>12</td>
      <td>8</td>
      <td>60</td>
      <td>3</td>
      <td>2-1</td>
      <td>67</td>
      <td>12</td>
      <td>6-6</td>
      <td>50</td>
      <td>14</td>
      <td>8-6</td>
      <td>57</td>
      <td>4</td>
      <td>3-1</td>
      <td>75</td>
      <td>4</td>
      <td>2-2</td>
      <td>50</td>
      <td>2</td>
      <td>0-2</td>
      <td>0</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lissandra</td>
      <td>39</td>
      <td>20</td>
      <td>19</td>
      <td>51</td>
      <td>19</td>
      <td>12</td>
      <td>7</td>
      <td>63</td>
      <td>20</td>
      <td>8</td>
      <td>12</td>
      <td>40</td>
      <td>7</td>
      <td>2-5</td>
      <td>29</td>
      <td>10</td>
      <td>5-5</td>
      <td>50</td>
      <td>14</td>
      <td>10-4</td>
      <td>71</td>
      <td>4</td>
      <td>0-4</td>
      <td>0</td>
      <td>4</td>
      <td>3-1</td>
      <td>75</td>
      <td>0</td>
      <td>0-0</td>
      <td>0</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ezreal</td>
      <td>38</td>
      <td>16</td>
      <td>22</td>
      <td>42</td>
      <td>11</td>
      <td>5</td>
      <td>6</td>
      <td>45</td>
      <td>27</td>
      <td>11</td>
      <td>16</td>
      <td>41</td>
      <td>2</td>
      <td>0-2</td>
      <td>0</td>
      <td>11</td>
      <td>4-7</td>
      <td>36</td>
      <td>17</td>
      <td>8-9</td>
      <td>47</td>
      <td>4</td>
      <td>2-2</td>
      <td>50</td>
      <td>4</td>
      <td>2-2</td>
      <td>50</td>
      <td>0</td>
      <td>0-0</td>
      <td>0</td>
      <td>Spring</td>
    </tr>
  </tbody>
</table>
</div>




```python
champion_data.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Champion 1</th>
      <th>∑ 2</th>
      <th>W 3</th>
      <th>L 4</th>
      <th>WR 5</th>
      <th>∑ 6</th>
      <th>W 7</th>
      <th>L 8</th>
      <th>WR 9</th>
      <th>∑ 10</th>
      <th>W 11</th>
      <th>L 12</th>
      <th>WR 13</th>
      <th>&lt;25 games</th>
      <th>&lt;25 winrate</th>
      <th>&lt;25 win percentage</th>
      <th>25-30 games</th>
      <th>25-30 winrate</th>
      <th>25-30 win percentage</th>
      <th>30-35 games</th>
      <th>30-35 winrate</th>
      <th>30-35 win percentage</th>
      <th>35-40 games</th>
      <th>35-40 winrate</th>
      <th>35-40 win percentage</th>
      <th>40-45 games</th>
      <th>40-45 winrate</th>
      <th>40-45 win percentage</th>
      <th>&gt;45 games</th>
      <th>&gt;45 winrate</th>
      <th>&gt;45 win percentage</th>
      <th>Split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>194</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194</td>
      <td>194.000000</td>
      <td>194.000000</td>
      <td>194</td>
      <td>194.000000</td>
      <td>194</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>114</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Trundle</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0-0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0-0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0-0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0-0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0-0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0-0</td>
      <td>NaN</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>88</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>40</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>87</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>148</td>
      <td>NaN</td>
      <td>99</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>11.701031</td>
      <td>5.850515</td>
      <td>5.850515</td>
      <td>48.463918</td>
      <td>5.850515</td>
      <td>3.092784</td>
      <td>2.757732</td>
      <td>44.541237</td>
      <td>5.850515</td>
      <td>2.757732</td>
      <td>3.092784</td>
      <td>37.314433</td>
      <td>1.185567</td>
      <td>NaN</td>
      <td>26.773196</td>
      <td>3.247423</td>
      <td>NaN</td>
      <td>39.768041</td>
      <td>3.762887</td>
      <td>NaN</td>
      <td>38.278351</td>
      <td>1.804124</td>
      <td>NaN</td>
      <td>27.273196</td>
      <td>1.391753</td>
      <td>NaN</td>
      <td>28.360825</td>
      <td>0.309278</td>
      <td>NaN</td>
      <td>12.541237</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>12.395608</td>
      <td>6.432054</td>
      <td>6.407033</td>
      <td>28.149697</td>
      <td>6.337104</td>
      <td>3.495615</td>
      <td>3.234860</td>
      <td>33.863653</td>
      <td>6.741975</td>
      <td>3.435274</td>
      <td>3.701512</td>
      <td>30.968296</td>
      <td>1.532582</td>
      <td>NaN</td>
      <td>38.953009</td>
      <td>3.726480</td>
      <td>NaN</td>
      <td>40.000295</td>
      <td>4.175609</td>
      <td>NaN</td>
      <td>35.023084</td>
      <td>2.466939</td>
      <td>NaN</td>
      <td>36.952183</td>
      <td>1.772307</td>
      <td>NaN</td>
      <td>37.910362</td>
      <td>0.608418</td>
      <td>NaN</td>
      <td>32.237475</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>33.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>6.000000</td>
      <td>3.500000</td>
      <td>3.000000</td>
      <td>50.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>50.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>40.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>33.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>17.000000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>64.750000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>4.750000</td>
      <td>63.750000</td>
      <td>8.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>55.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>50.000000</td>
      <td>5.000000</td>
      <td>NaN</td>
      <td>75.000000</td>
      <td>5.000000</td>
      <td>NaN</td>
      <td>62.750000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>50.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>50.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>54.000000</td>
      <td>27.000000</td>
      <td>27.000000</td>
      <td>100.000000</td>
      <td>28.000000</td>
      <td>13.000000</td>
      <td>17.000000</td>
      <td>100.000000</td>
      <td>37.000000</td>
      <td>16.000000</td>
      <td>21.000000</td>
      <td>100.000000</td>
      <td>8.000000</td>
      <td>NaN</td>
      <td>100.000000</td>
      <td>16.000000</td>
      <td>NaN</td>
      <td>100.000000</td>
      <td>21.000000</td>
      <td>NaN</td>
      <td>100.000000</td>
      <td>11.000000</td>
      <td>NaN</td>
      <td>100.000000</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>100.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>100.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### It looks like duplicate champions in this dataset is caused by data seperation caused by the split. This should not be a problem if we keep this in mind during our analysis.


```python
pd.concat(g for _, g in champion_data.groupby("Champion 1") if len(g) > 1).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Champion 1</th>
      <th>∑ 2</th>
      <th>W 3</th>
      <th>L 4</th>
      <th>WR 5</th>
      <th>∑ 6</th>
      <th>W 7</th>
      <th>L 8</th>
      <th>WR 9</th>
      <th>∑ 10</th>
      <th>W 11</th>
      <th>L 12</th>
      <th>WR 13</th>
      <th>&lt;25 games</th>
      <th>&lt;25 winrate</th>
      <th>&lt;25 win percentage</th>
      <th>25-30 games</th>
      <th>25-30 winrate</th>
      <th>25-30 win percentage</th>
      <th>30-35 games</th>
      <th>30-35 winrate</th>
      <th>30-35 win percentage</th>
      <th>35-40 games</th>
      <th>35-40 winrate</th>
      <th>35-40 win percentage</th>
      <th>40-45 games</th>
      <th>40-45 winrate</th>
      <th>40-45 win percentage</th>
      <th>&gt;45 games</th>
      <th>&gt;45 winrate</th>
      <th>&gt;45 win percentage</th>
      <th>Split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Aatrox</td>
      <td>30</td>
      <td>14</td>
      <td>16</td>
      <td>47</td>
      <td>20</td>
      <td>11</td>
      <td>9</td>
      <td>55</td>
      <td>10</td>
      <td>3</td>
      <td>7</td>
      <td>30</td>
      <td>3</td>
      <td>1-2</td>
      <td>33</td>
      <td>7</td>
      <td>3-4</td>
      <td>43</td>
      <td>7</td>
      <td>2-5</td>
      <td>29</td>
      <td>8</td>
      <td>4-4</td>
      <td>50</td>
      <td>4</td>
      <td>3-1</td>
      <td>75</td>
      <td>1</td>
      <td>1-0</td>
      <td>100</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Aatrox</td>
      <td>41</td>
      <td>16</td>
      <td>25</td>
      <td>39</td>
      <td>28</td>
      <td>11</td>
      <td>17</td>
      <td>39</td>
      <td>13</td>
      <td>5</td>
      <td>8</td>
      <td>38</td>
      <td>1</td>
      <td>0-1</td>
      <td>0</td>
      <td>9</td>
      <td>6-3</td>
      <td>67</td>
      <td>16</td>
      <td>8-8</td>
      <td>50</td>
      <td>10</td>
      <td>1-9</td>
      <td>10</td>
      <td>5</td>
      <td>1-4</td>
      <td>20</td>
      <td>0</td>
      <td>0-0</td>
      <td>0</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Akali</td>
      <td>12</td>
      <td>5</td>
      <td>7</td>
      <td>42</td>
      <td>8</td>
      <td>4</td>
      <td>4</td>
      <td>50</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>25</td>
      <td>3</td>
      <td>1-2</td>
      <td>33</td>
      <td>4</td>
      <td>2-2</td>
      <td>50</td>
      <td>4</td>
      <td>1-3</td>
      <td>25</td>
      <td>0</td>
      <td>0-0</td>
      <td>0</td>
      <td>1</td>
      <td>1-0</td>
      <td>100</td>
      <td>0</td>
      <td>0-0</td>
      <td>0</td>
      <td>Spring</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Akali</td>
      <td>43</td>
      <td>23</td>
      <td>20</td>
      <td>53</td>
      <td>22</td>
      <td>12</td>
      <td>10</td>
      <td>55</td>
      <td>21</td>
      <td>11</td>
      <td>10</td>
      <td>52</td>
      <td>5</td>
      <td>4-1</td>
      <td>80</td>
      <td>13</td>
      <td>4-9</td>
      <td>31</td>
      <td>13</td>
      <td>5-8</td>
      <td>38</td>
      <td>10</td>
      <td>8-2</td>
      <td>80</td>
      <td>2</td>
      <td>2-0</td>
      <td>100</td>
      <td>0</td>
      <td>0-0</td>
      <td>0</td>
      <td>Summer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alistar</td>
      <td>39</td>
      <td>21</td>
      <td>18</td>
      <td>54</td>
      <td>19</td>
      <td>9</td>
      <td>10</td>
      <td>47</td>
      <td>20</td>
      <td>12</td>
      <td>8</td>
      <td>60</td>
      <td>3</td>
      <td>2-1</td>
      <td>67</td>
      <td>12</td>
      <td>6-6</td>
      <td>50</td>
      <td>14</td>
      <td>8-6</td>
      <td>57</td>
      <td>4</td>
      <td>3-1</td>
      <td>75</td>
      <td>4</td>
      <td>2-2</td>
      <td>50</td>
      <td>2</td>
      <td>0-2</td>
      <td>0</td>
      <td>Spring</td>
    </tr>
  </tbody>
</table>
</div>



## Match Data


```python
match_data = raw_match_data.copy().drop('Unnamed: 0', axis=1)
match_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team1</th>
      <th>Team2</th>
      <th>Result</th>
      <th>UTC</th>
      <th>PBP</th>
      <th>Color</th>
      <th>MVP</th>
      <th>Blue</th>
      <th>Red</th>
      <th>Sel</th>
      <th>Choice</th>
      <th>Day</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>G2 Esports</td>
      <td>Splyce</td>
      <td>True</td>
      <td>2019-06-07</td>
      <td>Quickshot</td>
      <td>Ender</td>
      <td>Jankos</td>
      <td>G2</td>
      <td>SPY</td>
      <td>G2</td>
      <td>1</td>
      <td>Fri</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Team Vitality</td>
      <td>Origen</td>
      <td>False</td>
      <td>2019-06-07</td>
      <td>Quickshot</td>
      <td>Ender</td>
      <td>Patrik</td>
      <td>OG</td>
      <td>VIT</td>
      <td>VIT</td>
      <td>0</td>
      <td>Fri</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rogue</td>
      <td>Misfits Gaming</td>
      <td>False</td>
      <td>2019-06-07</td>
      <td>Drakos</td>
      <td>Froskurinn</td>
      <td>Hans Sama</td>
      <td>RGE</td>
      <td>MSF</td>
      <td>RGE</td>
      <td>1</td>
      <td>Fri</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fnatic</td>
      <td>SK Gaming</td>
      <td>True</td>
      <td>2019-06-07</td>
      <td>Drakos</td>
      <td>Froskurinn</td>
      <td>Rekkles</td>
      <td>FNC</td>
      <td>SK</td>
      <td>FNC</td>
      <td>1</td>
      <td>Fri</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FC Schalke 04</td>
      <td>Excel Esports</td>
      <td>True</td>
      <td>2019-06-07</td>
      <td>Drakos</td>
      <td>Froskurinn</td>
      <td>Trick</td>
      <td>S04</td>
      <td>XL</td>
      <td>S04</td>
      <td>1</td>
      <td>Fri</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
match_data = match_data.drop('Time', axis=1)
```


```python
match_data.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team1</th>
      <th>Team2</th>
      <th>Result</th>
      <th>UTC</th>
      <th>PBP</th>
      <th>Color</th>
      <th>MVP</th>
      <th>Blue</th>
      <th>Red</th>
      <th>Sel</th>
      <th>Choice</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>182</td>
      <td>182</td>
      <td>182</td>
      <td>182</td>
      <td>182</td>
      <td>182</td>
      <td>180</td>
      <td>182</td>
      <td>182</td>
      <td>182</td>
      <td>182.000000</td>
      <td>182</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>10</td>
      <td>10</td>
      <td>2</td>
      <td>36</td>
      <td>3</td>
      <td>16</td>
      <td>51</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>SK Gaming</td>
      <td>Splyce</td>
      <td>False</td>
      <td>2019-03-16</td>
      <td>Medic</td>
      <td>Vedius</td>
      <td>Jankos</td>
      <td>G2</td>
      <td>VIT</td>
      <td>SK</td>
      <td>NaN</td>
      <td>Sat</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>24</td>
      <td>22</td>
      <td>101</td>
      <td>6</td>
      <td>71</td>
      <td>53</td>
      <td>11</td>
      <td>25</td>
      <td>24</td>
      <td>24</td>
      <td>NaN</td>
      <td>92</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.730769</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.444784</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Player Data


```python
player_data = raw_player_data.copy().drop('Unnamed: 0', axis=1)
player_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Team</th>
      <th>Pos</th>
      <th>GP</th>
      <th>W%</th>
      <th>K</th>
      <th>D</th>
      <th>A</th>
      <th>KDA</th>
      <th>KP</th>
      <th>DTH%</th>
      <th>FB%</th>
      <th>GD10</th>
      <th>XPD10</th>
      <th>CSD10</th>
      <th>CSPM</th>
      <th>CS%P15</th>
      <th>DPM</th>
      <th>DMG%</th>
      <th>EGPM</th>
      <th>Gold%</th>
      <th>WPM</th>
      <th>WCPM</th>
      <th>Split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abbedagge</td>
      <td>Schalke 04</td>
      <td>Middle</td>
      <td>19.0</td>
      <td>47.0</td>
      <td>41.0</td>
      <td>43.0</td>
      <td>79.0</td>
      <td>2.8</td>
      <td>63.2</td>
      <td>21.9</td>
      <td>11.0</td>
      <td>-115.0</td>
      <td>-125.0</td>
      <td>-3.3</td>
      <td>9.0</td>
      <td>26.6</td>
      <td>417.0</td>
      <td>24.6</td>
      <td>257.9</td>
      <td>22.5</td>
      <td>0.51</td>
      <td>0.19</td>
      <td>Spring Regular</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alphari</td>
      <td>Origen</td>
      <td>Top</td>
      <td>18.0</td>
      <td>67.0</td>
      <td>29.0</td>
      <td>30.0</td>
      <td>73.0</td>
      <td>3.4</td>
      <td>51.8</td>
      <td>19.4</td>
      <td>11.0</td>
      <td>133.0</td>
      <td>36.0</td>
      <td>1.7</td>
      <td>8.0</td>
      <td>22.4</td>
      <td>354.0</td>
      <td>21.8</td>
      <td>243.5</td>
      <td>20.9</td>
      <td>0.54</td>
      <td>0.15</td>
      <td>Spring Regular</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Attila</td>
      <td>Vitality</td>
      <td>ADC</td>
      <td>18.0</td>
      <td>56.0</td>
      <td>48.0</td>
      <td>30.0</td>
      <td>77.0</td>
      <td>4.2</td>
      <td>60.1</td>
      <td>14.9</td>
      <td>11.0</td>
      <td>-112.0</td>
      <td>-22.0</td>
      <td>-3.9</td>
      <td>10.2</td>
      <td>33.4</td>
      <td>458.0</td>
      <td>25.6</td>
      <td>316.7</td>
      <td>26.9</td>
      <td>0.35</td>
      <td>0.39</td>
      <td>Spring Regular</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Broxah</td>
      <td>Fnatic</td>
      <td>Jungle</td>
      <td>18.0</td>
      <td>61.0</td>
      <td>49.0</td>
      <td>27.0</td>
      <td>93.0</td>
      <td>5.3</td>
      <td>66.7</td>
      <td>15.5</td>
      <td>33.0</td>
      <td>160.0</td>
      <td>33.0</td>
      <td>-1.8</td>
      <td>5.0</td>
      <td>13.2</td>
      <td>255.0</td>
      <td>14.9</td>
      <td>200.1</td>
      <td>17.1</td>
      <td>0.36</td>
      <td>0.34</td>
      <td>Spring Regular</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bwipo</td>
      <td>Fnatic</td>
      <td>Top</td>
      <td>18.0</td>
      <td>61.0</td>
      <td>41.0</td>
      <td>42.0</td>
      <td>80.0</td>
      <td>2.9</td>
      <td>56.8</td>
      <td>24.1</td>
      <td>17.0</td>
      <td>-275.0</td>
      <td>-133.0</td>
      <td>-10.0</td>
      <td>7.6</td>
      <td>23.6</td>
      <td>444.0</td>
      <td>25.5</td>
      <td>239.7</td>
      <td>20.4</td>
      <td>0.48</td>
      <td>0.16</td>
      <td>Spring Regular</td>
    </tr>
  </tbody>
</table>
</div>




```python
player_data.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Team</th>
      <th>Pos</th>
      <th>GP</th>
      <th>W%</th>
      <th>K</th>
      <th>D</th>
      <th>A</th>
      <th>KDA</th>
      <th>KP</th>
      <th>DTH%</th>
      <th>FB%</th>
      <th>GD10</th>
      <th>XPD10</th>
      <th>CSD10</th>
      <th>CSPM</th>
      <th>CS%P15</th>
      <th>DPM</th>
      <th>DMG%</th>
      <th>EGPM</th>
      <th>Gold%</th>
      <th>WPM</th>
      <th>WCPM</th>
      <th>Split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>221</td>
      <td>221</td>
      <td>221</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221.000000</td>
      <td>221</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>74</td>
      <td>10</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Caps</td>
      <td>Fnatic</td>
      <td>Middle</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Summer Regular</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>7</td>
      <td>32</td>
      <td>49</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>79</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.769231</td>
      <td>47.669683</td>
      <td>27.909502</td>
      <td>27.954751</td>
      <td>60.312217</td>
      <td>3.748416</td>
      <td>62.538914</td>
      <td>19.767421</td>
      <td>22.565611</td>
      <td>-36.751131</td>
      <td>-23.180995</td>
      <td>-0.627149</td>
      <td>6.389593</td>
      <td>19.987330</td>
      <td>369.900452</td>
      <td>19.955204</td>
      <td>232.807240</td>
      <td>19.959276</td>
      <td>0.674525</td>
      <td>0.262172</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.587602</td>
      <td>28.618405</td>
      <td>22.987960</td>
      <td>17.402658</td>
      <td>42.786226</td>
      <td>3.470874</td>
      <td>8.713064</td>
      <td>5.881460</td>
      <td>19.179096</td>
      <td>369.681529</td>
      <td>274.355797</td>
      <td>9.908811</td>
      <td>3.125617</td>
      <td>10.246011</td>
      <td>162.910458</td>
      <td>7.724985</td>
      <td>75.567291</td>
      <td>6.083156</td>
      <td>0.394603</td>
      <td>0.115297</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>17.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-2312.000000</td>
      <td>-1311.000000</td>
      <td>-53.000000</td>
      <td>0.700000</td>
      <td>2.800000</td>
      <td>101.000000</td>
      <td>5.400000</td>
      <td>83.700000</td>
      <td>8.900000</td>
      <td>0.190000</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.000000</td>
      <td>30.000000</td>
      <td>8.000000</td>
      <td>12.000000</td>
      <td>24.000000</td>
      <td>2.200000</td>
      <td>57.700000</td>
      <td>16.300000</td>
      <td>9.000000</td>
      <td>-129.000000</td>
      <td>-133.000000</td>
      <td>-3.900000</td>
      <td>4.400000</td>
      <td>11.200000</td>
      <td>236.000000</td>
      <td>12.700000</td>
      <td>176.200000</td>
      <td>15.300000</td>
      <td>0.430000</td>
      <td>0.170000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.000000</td>
      <td>50.000000</td>
      <td>22.000000</td>
      <td>28.000000</td>
      <td>57.000000</td>
      <td>3.000000</td>
      <td>64.100000</td>
      <td>20.000000</td>
      <td>20.000000</td>
      <td>-17.000000</td>
      <td>-16.000000</td>
      <td>-0.400000</td>
      <td>7.700000</td>
      <td>23.100000</td>
      <td>389.000000</td>
      <td>22.500000</td>
      <td>242.900000</td>
      <td>21.400000</td>
      <td>0.480000</td>
      <td>0.250000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.000000</td>
      <td>61.000000</td>
      <td>43.000000</td>
      <td>42.000000</td>
      <td>89.000000</td>
      <td>4.200000</td>
      <td>68.400000</td>
      <td>23.500000</td>
      <td>33.000000</td>
      <td>106.000000</td>
      <td>100.000000</td>
      <td>3.400000</td>
      <td>8.900000</td>
      <td>28.100000</td>
      <td>467.000000</td>
      <td>26.100000</td>
      <td>287.300000</td>
      <td>25.000000</td>
      <td>0.760000</td>
      <td>0.340000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.000000</td>
      <td>100.000000</td>
      <td>104.000000</td>
      <td>70.000000</td>
      <td>188.000000</td>
      <td>40.000000</td>
      <td>83.300000</td>
      <td>50.000000</td>
      <td>100.000000</td>
      <td>2312.000000</td>
      <td>973.000000</td>
      <td>68.000000</td>
      <td>11.000000</td>
      <td>40.500000</td>
      <td>1186.000000</td>
      <td>47.900000</td>
      <td>500.400000</td>
      <td>32.800000</td>
      <td>1.800000</td>
      <td>0.560000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### It looks like the reason for the 74 unique values with 221 count value is because players played multiple positions and the different splits.  


```python
pd.concat(g for _, g in player_data.groupby("Player") if len(g) > 1).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Team</th>
      <th>Pos</th>
      <th>GP</th>
      <th>W%</th>
      <th>K</th>
      <th>D</th>
      <th>A</th>
      <th>KDA</th>
      <th>KP</th>
      <th>DTH%</th>
      <th>FB%</th>
      <th>GD10</th>
      <th>XPD10</th>
      <th>CSD10</th>
      <th>CSPM</th>
      <th>CS%P15</th>
      <th>DPM</th>
      <th>DMG%</th>
      <th>EGPM</th>
      <th>Gold%</th>
      <th>WPM</th>
      <th>WCPM</th>
      <th>Split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abbedagge</td>
      <td>Schalke 04</td>
      <td>Middle</td>
      <td>19.0</td>
      <td>47.0</td>
      <td>41.0</td>
      <td>43.0</td>
      <td>79.0</td>
      <td>2.8</td>
      <td>63.2</td>
      <td>21.9</td>
      <td>11.0</td>
      <td>-115.0</td>
      <td>-125.0</td>
      <td>-3.3</td>
      <td>9.0</td>
      <td>26.6</td>
      <td>417.0</td>
      <td>24.6</td>
      <td>257.9</td>
      <td>22.5</td>
      <td>0.51</td>
      <td>0.19</td>
      <td>Spring Regular</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Abbedagge</td>
      <td>Schalke 04</td>
      <td>Middle</td>
      <td>18.0</td>
      <td>61.0</td>
      <td>65.0</td>
      <td>43.0</td>
      <td>115.0</td>
      <td>4.2</td>
      <td>65.5</td>
      <td>22.1</td>
      <td>11.0</td>
      <td>-60.0</td>
      <td>159.0</td>
      <td>-1.3</td>
      <td>8.8</td>
      <td>26.4</td>
      <td>509.0</td>
      <td>27.7</td>
      <td>284.9</td>
      <td>24.2</td>
      <td>0.46</td>
      <td>0.20</td>
      <td>Summer Regular</td>
    </tr>
    <tr>
      <th>171</th>
      <td>Abbedagge</td>
      <td>Schalke 04</td>
      <td>Middle</td>
      <td>11.0</td>
      <td>55.0</td>
      <td>50.0</td>
      <td>31.0</td>
      <td>63.0</td>
      <td>3.6</td>
      <td>59.5</td>
      <td>21.4</td>
      <td>36.0</td>
      <td>-30.0</td>
      <td>169.0</td>
      <td>-1.4</td>
      <td>8.7</td>
      <td>27.5</td>
      <td>532.0</td>
      <td>27.2</td>
      <td>288.3</td>
      <td>25.1</td>
      <td>0.46</td>
      <td>0.26</td>
      <td>Summer Playoffs</td>
    </tr>
    <tr>
      <th>201</th>
      <td>Abbedagge</td>
      <td>Schalke 04</td>
      <td>Middle</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>0.9</td>
      <td>44.8</td>
      <td>26.9</td>
      <td>0.0</td>
      <td>-671.0</td>
      <td>-177.0</td>
      <td>-12.7</td>
      <td>7.1</td>
      <td>23.6</td>
      <td>396.0</td>
      <td>25.5</td>
      <td>200.1</td>
      <td>20.5</td>
      <td>0.53</td>
      <td>0.06</td>
      <td>Gauntlet</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alphari</td>
      <td>Origen</td>
      <td>Top</td>
      <td>18.0</td>
      <td>67.0</td>
      <td>29.0</td>
      <td>30.0</td>
      <td>73.0</td>
      <td>3.4</td>
      <td>51.8</td>
      <td>19.4</td>
      <td>11.0</td>
      <td>133.0</td>
      <td>36.0</td>
      <td>1.7</td>
      <td>8.0</td>
      <td>22.4</td>
      <td>354.0</td>
      <td>21.8</td>
      <td>243.5</td>
      <td>20.9</td>
      <td>0.54</td>
      <td>0.15</td>
      <td>Spring Regular</td>
    </tr>
  </tbody>
</table>
</div>



# Analysis

## Average Damage Done by Role


```python
damage_data = player_data.copy()
```


```python
damage_data['Pos'].unique()
```




    array(['Middle', 'Top', 'ADC', 'Jungle', 'Support'], dtype=object)




```python
dmg = []
for i in damage_data['Pos'].unique():
    dmg_by_role = damage_data['Pos'] == i
    dmg_by_role = damage_data[dmg_by_role]
    dmg_by_role = dmg_by_role['DMG%']
    dmg.append(dmg_by_role.aggregate('mean'))
```


```python
dmg = {'Dmg%':dmg, 'Role':damage_data['Pos'].unique()}
df = pd.DataFrame(data=dmg)
df.set_index('Role')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dmg%</th>
    </tr>
    <tr>
      <th>Role</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Middle</th>
      <td>26.218367</td>
    </tr>
    <tr>
      <th>Top</th>
      <td>23.527907</td>
    </tr>
    <tr>
      <th>ADC</th>
      <td>25.925000</td>
    </tr>
    <tr>
      <th>Jungle</th>
      <td>14.305128</td>
    </tr>
    <tr>
      <th>Support</th>
      <td>9.023913</td>
    </tr>
  </tbody>
</table>
</div>



#### This chart contains results that were around what was expected, it is interesting how Jungle is such an agressive role but because of the play style in pro play junglers tend to not have high Dmg% contribution. 


```python
sns.barplot(x=df['Dmg%'], y=df['Role'])
plt.title('Average Damage Grouped by Role')
```




    Text(0.5, 1.0, 'Average Damage Grouped by Role')




![png](LEC%202019%20Exploratory%20Analysis_files/LEC%202019%20Exploratory%20Analysis_25_1.png)


## Average Damage Done by Players


```python
team_data = player_data.copy()
```


```python
team_data['Team'].unique()
```




    array(['Schalke 04', 'Origen', 'Vitality', 'Fnatic', 'Excel Esports',
           'G2 Esports', 'SK Gaming', 'Misfits', 'Rogue', 'Splyce'],
          dtype=object)




```python
dmg = []
for i in team_data['Team'].unique():
    dmg_by_team = team_data['Team'] == i
    dmg_by_team = team_data[dmg_by_team]
    dmg_by_team = dmg_by_team['DMG%']
    dmg.append(dmg_by_team.aggregate('mean'))
```


```python
dmg = {'Dmg%':dmg, 'Team':team_data['Team'].unique()}
df = pd.DataFrame(data=dmg)
df.set_index('Team')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dmg%</th>
    </tr>
    <tr>
      <th>Team</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Schalke 04</th>
      <td>19.995000</td>
    </tr>
    <tr>
      <th>Origen</th>
      <td>19.995000</td>
    </tr>
    <tr>
      <th>Vitality</th>
      <td>20.654545</td>
    </tr>
    <tr>
      <th>Fnatic</th>
      <td>20.428125</td>
    </tr>
    <tr>
      <th>Excel Esports</th>
      <td>19.950000</td>
    </tr>
    <tr>
      <th>G2 Esports</th>
      <td>19.136667</td>
    </tr>
    <tr>
      <th>SK Gaming</th>
      <td>20.312500</td>
    </tr>
    <tr>
      <th>Misfits</th>
      <td>19.037500</td>
    </tr>
    <tr>
      <th>Rogue</th>
      <td>19.936842</td>
    </tr>
    <tr>
      <th>Splyce</th>
      <td>20.026923</td>
    </tr>
  </tbody>
</table>
</div>



#### For me this chart was a little surprising, seeing that G2 Esports is among the teams that did the least amount of damage. Which may indicate that G2 a known agressive team might have some players that may be dealing lower amount of damage than the average.


```python
sns.barplot(x=df['Dmg%'], y=df['Team'])
plt.title('Average Damage Grouped by Team')
```




    Text(0.5, 1.0, 'Average Damage Grouped by Team')




![png](LEC%202019%20Exploratory%20Analysis_files/LEC%202019%20Exploratory%20Analysis_32_1.png)


#### Looking further into this it seems like G2 really does have a player that has the lowest DMG% contribution and it turns out to be Caps on ADC which explains the low DMG% contribution. Caps is primarily a mid laner as such this number can be explained.


```python
dmg_by_team = team_data['Team'] == 'G2 Esports'
dmg_by_team = team_data[dmg_by_team]
dmg_by_team = dmg_by_team[['Team', 'Player', 'DMG%', 'Pos']]
dmg_by_team.aggregate(['min', 'max']).rename(index={'min':'min_g2', 'max':'max_g2'}).append(team_data[['Team', 'Player', 'DMG%', 'Pos']]
                                             .aggregate(['min', 'max'])
                                             .rename(index={'min':'min_all', 'max':'max_all'}))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Player</th>
      <th>DMG%</th>
      <th>Pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min_g2</th>
      <td>G2 Esports</td>
      <td>Caps</td>
      <td>5.4</td>
      <td>ADC</td>
    </tr>
    <tr>
      <th>max_g2</th>
      <td>G2 Esports</td>
      <td>promisq</td>
      <td>29.5</td>
      <td>Top</td>
    </tr>
    <tr>
      <th>min_all</th>
      <td>Excel Esports</td>
      <td>Abbedagge</td>
      <td>5.4</td>
      <td>ADC</td>
    </tr>
    <tr>
      <th>max_all</th>
      <td>Vitality</td>
      <td>sOAZ</td>
      <td>47.9</td>
      <td>Top</td>
    </tr>
  </tbody>
</table>
</div>



## Side Selection


```python
side_selection = match_data.copy()
```


```python
side_selection['Choice'].value_counts()
```




    1    133
    0     49
    Name: Choice, dtype: int64




```python
labels = 'Blue', 'Red'
choice = side_selection['Choice'].value_counts()
colors = ['xkcd:azure', 'crimson']
explode = (0.1, 0)  

plt.pie(choice, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=False, startangle=45, textprops={'fontsize': 14})

plt.title("Side Selection", fontsize=16)
plt.show()
```


![png](LEC%202019%20Exploratory%20Analysis_files/LEC%202019%20Exploratory%20Analysis_38_0.png)



```python
win_with_selec = side_selection.loc[:, ('Team1', 'Team2', 'Sel', 'Choice', 'Result')]
win_with_selec.loc[:, ('Result')] = side_selection.loc[:, ('Result')].map({True:1, False:0})

win_with_selec = 1 == win_with_selec['Result']
win_with_selec = side_selection[win_with_selec]
```

#### As Blue side is the favorable side to pick it is no surprise that teams won more by picking it. However the data also shows that teams opted to picking red side in some cases. 


```python
 win_with_selec['Choice'].value_counts()
```




    1    63
    0    18
    Name: Choice, dtype: int64




```python
labels = 'Blue', 'Red'
choice = win_with_selec['Choice'].value_counts()
colors = ['xkcd:azure', 'crimson']
explode = (0.1, 0)  

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.pie(choice, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=False, startangle=45, textprops={'fontsize': 14})
ax1.set_title("Teams that Won with Side Choice", fontsize=16)

ax2.table(cellText=[['63'],["18"], ['81']], rowLabels=['Blue', 'Red', 'Total'], loc='right')
ax2.axis('off')

plt.show()
```


![png](LEC%202019%20Exploratory%20Analysis_files/LEC%202019%20Exploratory%20Analysis_42_0.png)



```python
win_without_selec = side_selection.loc[:, ('Team1', 'Team2', 'Sel', 'Choice', 'Result')]
win_without_selec.loc[:, ('Result')] = side_selection.loc[:, ('Result')].map({True:1, False:0})

win_without_selec = 0 == win_without_selec['Result']
win_without_selec = side_selection[win_without_selec]
```


```python
table_df = pd.DataFrame(data=win_without_selec['Choice'].map({1:0, 0:1}).value_counts())
table_df = table_df.rename(index={0:'Red', 1:'Blue'})

table_df.index.name = "Side"
```

#### This chart actually shows the opposite of what we found in the previous chart. While the majority of the teams here picked Blue side, the team on Red side seems to have a significantly more wins. This indicates that although Blue side may give teams a stragtegic advantage it is not significant enough to decide games. 


```python
labels = 'Red', 'Blue'
choice = win_without_selec['Choice'].value_counts()
colors = ['crimson', 'xkcd:azure']
explode = (0.1, 0)  

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.pie(choice, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=False, startangle=45, textprops={'fontsize': 14})
ax1.set_title("Teams that Won without Side Choice", fontsize=16)

ax2.table(cellText=[['70'],["31"], ['101']], rowLabels=['Red', 'Blue', 'Total'], loc='right', snap=True)
ax2.axis('off')


plt.show()
```


![png](LEC%202019%20Exploratory%20Analysis_files/LEC%202019%20Exploratory%20Analysis_46_0.png)


## Day of the week affect


```python
day_effect = match_data.copy()
team_names = []

for i, j, k in zip(day_effect['Result'], day_effect['Team1'], day_effect['Team2']):
    if i:
        team_names.append(j)
    else:
        team_names.append(k)
        
day_effect['team_that_won'] = team_names
```


```python
day_effect['team_that_won'].unique()
```




    array(['G2 Esports', 'Origen', 'Misfits Gaming', 'Fnatic',
           'FC Schalke 04', 'Rogue', 'SK Gaming', 'Splyce', 'Team Vitality',
           'Excel Esports'], dtype=object)



#### This data was mostly plotted out of curiosity to see if there was any teams that performed better on a certain day of the week. Interpret this graph as you will. 


```python
team_win = []
fig, axs = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
plot_data = []

for i, j in zip(day_effect['team_that_won'], day_effect['team_that_won'].unique()):
    team_win = day_effect['team_that_won'] == j
    team_win = day_effect[team_win]
    x = team_win['Day'].value_counts().index
    y = team_win['Day'].value_counts().values
    plot_data.append((x,y, j))
    
i = 0
k = 0
for j in plot_data:
    axs[i,k].bar(x=j[0], height=j[1], color=np.random.rand(1,3))
    axs[i, k].set_title(j[2])
    k += 1
    if(k > 4):
        i += 1
        k = 0   
```


![png](LEC%202019%20Exploratory%20Analysis_files/LEC%202019%20Exploratory%20Analysis_51_0.png)


## Champion Statistics

#### For the win rates I only considered champions that were played more than 5 times

### Average Champion Win Rate


```python
champ_stats = champion_data.copy()

avg_win_rate = champ_stats['∑ 2'] > 5
avg_win_rate = champ_stats[avg_win_rate]
avg_win_rate.sort_values(by=['WR 5'], ascending=False).head(10).reset_index(drop=True).loc[:, ('Champion 1', 'W 3', 'L 4', 'WR 5')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Champion 1</th>
      <th>W 3</th>
      <th>L 4</th>
      <th>WR 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Skarner</td>
      <td>7</td>
      <td>0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jax</td>
      <td>7</td>
      <td>1</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Vayne</td>
      <td>12</td>
      <td>2</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Swain</td>
      <td>5</td>
      <td>1</td>
      <td>83</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Olaf</td>
      <td>13</td>
      <td>4</td>
      <td>76</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kled</td>
      <td>8</td>
      <td>3</td>
      <td>73</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Xayah</td>
      <td>8</td>
      <td>3</td>
      <td>73</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Rakan</td>
      <td>15</td>
      <td>6</td>
      <td>71</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Qiyana</td>
      <td>13</td>
      <td>6</td>
      <td>68</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Varus</td>
      <td>4</td>
      <td>2</td>
      <td>67</td>
    </tr>
  </tbody>
</table>
</div>



### Chamption Win Rate Spring


```python
spring = champ_stats['Split'] == 'Spring'
spring = champ_stats[spring]
spring = champ_stats['∑ 6'] > 5
spring = champ_stats[spring]
spring.sort_values(by=['WR 9'], ascending=False).head(10).reset_index(drop=True).loc[:, ('Champion 1', 'W 7', 'L 8', 'WR 9')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Champion 1</th>
      <th>W 7</th>
      <th>L 8</th>
      <th>WR 9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Xayah</td>
      <td>6</td>
      <td>0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rakan</td>
      <td>10</td>
      <td>0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Syndra</td>
      <td>6</td>
      <td>1</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Olaf</td>
      <td>8</td>
      <td>2</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LeBlanc</td>
      <td>8</td>
      <td>2</td>
      <td>80</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Elise</td>
      <td>6</td>
      <td>2</td>
      <td>75</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Morgana</td>
      <td>6</td>
      <td>2</td>
      <td>75</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Irelia</td>
      <td>6</td>
      <td>2</td>
      <td>75</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Lee Sin</td>
      <td>9</td>
      <td>3</td>
      <td>75</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Kled</td>
      <td>5</td>
      <td>2</td>
      <td>71</td>
    </tr>
  </tbody>
</table>
</div>



### Chamption Win Rate Summer


```python
summer = champ_stats['Split'] == 'Summer'
summer = champ_stats[summer]
summer = champ_stats['∑ 10'] > 5
summer = champ_stats[summer]
summer.sort_values(by=['WR 13'], ascending=False).head(10).reset_index(drop=True).loc[:, ('Champion 1', 'W 11', 'L 12', 'WR 13')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Champion 1</th>
      <th>W 11</th>
      <th>L 12</th>
      <th>WR 13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jax</td>
      <td>6</td>
      <td>1</td>
      <td>86</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Vayne</td>
      <td>8</td>
      <td>2</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Olaf</td>
      <td>5</td>
      <td>2</td>
      <td>71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jayce</td>
      <td>5</td>
      <td>2</td>
      <td>71</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Qiyana</td>
      <td>7</td>
      <td>3</td>
      <td>70</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Karma</td>
      <td>7</td>
      <td>3</td>
      <td>70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Corki</td>
      <td>11</td>
      <td>5</td>
      <td>69</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Tahm Kench</td>
      <td>13</td>
      <td>6</td>
      <td>68</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jarvan IV</td>
      <td>12</td>
      <td>6</td>
      <td>67</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poppy</td>
      <td>4</td>
      <td>2</td>
      <td>67</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
