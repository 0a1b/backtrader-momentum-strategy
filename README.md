# backtrader-momentum-strategy
Momentum Strategy implemented in Backtrader for newbies

Since it is based on Teddy Kokers Momentum Strategy as well as two Backtrader blog posts, I would like to express my thankfulness to Teddy Koker and the creator of Backtrader, Daniel Rodriguez!

* https://teddykoker.com/2019/05/momentum-strategy-from-stocks-on-the-move-in-python/
* https://www.backtrader.com/blog/2019-07-19-rebalancing-conservative/rebalancing-conservative/
* https://www.backtrader.com/blog/2019-05-20-momentum-strategy/momentum-strategy/

Features:
- momentum strategy tweaked with respect to
  + Rate of Change (relative increase in given time)
  + time interval of trading (weekly, monthly etc.)
  + trailing stop loss
  + change the number of stocks selected from the universe
  + excel output of all stock over time rankings
- working pyfolio
- fundatmental screening (revenue growth over x years or P/S / revenue growth (PSG)
- working quantstat
- working multiprocessing (works only in linux)
- working heatmap which can depict different variable combinations & investment performance KPIs, both chosen by the user
- working optunity implementation
- working walk-forward-analysis to better get a sense of the degree of overoptimisation

disclaimer: working at one point of time... if I broke it in between, please leave a note.


In the beginning i didn't like to use it but some grafical features, i.e. pyfolio, only work within Jupyter so please do not give up here.
I will add a series of versions and each of them works according to what I remeber. I think this might help to understand the features I added later on better.

Help needed:
If someone manages to get this data to work: https://github.com/teddykoker/quant/tree/master/survivorship-free/data within the provided code, I would be super happy to know how. I haven't tried to add additional informaiton to the data so backtrader knows how long each symbol can be used, but hopefully there is an easier way. I tried to make it work in the "Sur" strategy, but failed big time.

Since sharing ist caring:
Here the super lazy version (haven't tested multiprocess on Colab):
https://colab.research.google.com/drive/1VgHUZNRFZNwoA-wvWvC3KxQMhnvI5xVR?usp=sharing
One just has to copy it and set up the folder structure.
