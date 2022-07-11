import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from binance.client import Client
from tinydb import TinyDB
import streamlit as st

# Initializing Binance
client = Client('', '')

coin_list = ['aave', 'ada', 'algo', 'atom', 'avax', 'axs', 'bat', 'bch', 'btc', 'chz', 'comp',
             'crv', 'doge', 'dot', 'enj', 'etc', 'eth', 'fil', 'gala', 'grt', 'icp', 'link', 'lrc', 'ltc', 'mana',
             'matic',
             'mkr', 'omg', 'shib', 'snx', 'sol', 'storj', 'sushi', 'xlm', 'xtz', 'zec']

chart_list = ['Order Flow', 'Volume Profile']

timeframe_list = ['5min', '15min', '1hr']

profile_list = ['Daily', 'Weekly']

st.set_page_config(layout="wide")

hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title('BTH Charts')

st.markdown(' ')

st.sidebar.title('Filters')

symbol = st.sidebar.selectbox(options=[coin.upper() for coin in coin_list], label='Symbol')

chart = st.sidebar.selectbox(options=[chart for chart in chart_list], label='Chart Type')

if chart == 'Order Flow':
    timeframe = st.sidebar.selectbox(options=[timeframe for timeframe in timeframe_list], label='Timeframe')
    st.markdown(f"<h3 style='text-align: left;'>{timeframe} {chart} of {symbol}</h3>",
                unsafe_allow_html=True)

elif chart == 'Volume Profile':
    timeframe = st.sidebar.selectbox(options=[timeframe for timeframe in profile_list], label='Timeframe')

db = TinyDB(f'Data/{symbol.upper()}_DERIV_Data.json')

if chart == 'Volume Profile' and timeframe == 'Daily':
  
    st.markdown(f"<h3 style='text-align: left;'>Previous Day's {chart} of {symbol}</h3>",
                unsafe_allow_html=True)
    
    try:

      df = pd.DataFrame.from_dict(db)

      df['Date'] = pd.to_datetime(df['Date'])

      daily_vp = df.copy()

      daily_vp['Volume'] = daily_vp['xAsks'] + daily_vp['xBids']

      binancetime = datetime.utcfromtimestamp(client.get_server_time()['serverTime'] / 1000)

      now = binancetime.replace(hour=0, minute=0, second=0, microsecond=0)
      prev_day = now - timedelta(days=1)

      now = now - timedelta(minutes=5)

      start = prev_day

      end = now

      daily_vp['Date'] = pd.to_datetime(daily_vp['Date'])

      daily_vp.set_index('Date', inplace=True)

      daily_vp = daily_vp.loc[start:end]

      today = df.copy()

      today['Date'] = pd.to_datetime(today['Date'])

      day_start = binancetime.replace(hour=0, minute=0, second=0, microsecond=0)

      today.set_index('Date', inplace=True)

      today = today.loc[day_start:]

      # Volume Profile 1Day
      bucket_size = 0.002 * max(daily_vp['Close'])
      volprofile = daily_vp['Volume'].groupby(
          daily_vp['Close'].apply(lambda x: bucket_size * round(x / bucket_size, 0))).sum()
      VPOC = volprofile.max()

      poc = volprofile.idxmax()
      volprofile_copy = volprofile.copy()
      total_volume = volprofile_copy.sum()
      value_area_volume = total_volume * 0.70

      val = 0
      vah = 0
      while value_area_volume >= 0:
          for x in range(len(volprofile_copy)):
              if volprofile_copy.values[x] == VPOC:
                  two_above = volprofile.values[x - 1] + volprofile_copy.values[x - 2] if x - 2 >= 0 else 0
                  two_below = volprofile.values[x + 1] + volprofile_copy.values[x + 2] if x + 2 <= len(
                      volprofile_copy) - 1 else 0
                  val = volprofile_copy.index[x - 1] if x - 1 >= 0 else val
                  vah = volprofile_copy.index[x + 1] if x + 1 <= len(volprofile_copy) - 1 else vah
                  if two_above >= two_below:
                      volprofile_copy.drop([volprofile_copy.index[x - 1], volprofile_copy.index[x - 2]],
                                           inplace=True)
                      value_area_volume = value_area_volume - two_above
                      break
                  else:
                      volprofile_copy.drop([volprofile_copy.index[x + 1], volprofile_copy.index[x + 2]],
                                           inplace=True)
                      value_area_volume = value_area_volume - two_below
                      break

      vah_value = round(vah, 3)
      val_value = round(val, 3)
      poc_value = round(poc, 3)

      vah_text = '{0:,}'.format(vah_value)
      val_text = '{0:,}'.format(val_value)
      poc_text = '{0:,}'.format(poc_value)

      vah_text = str(vah_text)
      val_text = str(val_text)
      poc_text = str(poc_text)

      current_chart = pd.concat([daily_vp, today], axis=0)

      fig1 = go.Candlestick(
          x=current_chart.index,
          open=current_chart.Open,
          high=current_chart.High,
          low=current_chart.Low,
          close=current_chart.Close,
          xaxis='x',
          yaxis='y',
          visible=True,
          showlegend=False,
          increasing_fillcolor='#24A06B',
          decreasing_fillcolor="#CC2E3C",
          increasing_line_color='#2EC886',
          decreasing_line_color='#FF3A4C',
          line=dict(width=1), opacity=1)

      fig2 = go.Bar(
          x=volprofile.values,
          y=volprofile.keys().values,
          orientation='h',
          xaxis='x2',
          yaxis='y',
          visible=True,
          showlegend=False,
          name='Volume Bars',
          marker_color='dodgerblue',
          opacity=0.2,
          text=volprofile.values,
          textposition='auto'
      )

      low = min(current_chart['Low'])
      high = max(current_chart['High'])
      layout = go.Layout(
          title=go.layout.Title(text="Volume Profile"),
          xaxis=go.layout.XAxis(
              side="bottom",
              title="Date",
              rangeslider=go.layout.xaxis.Rangeslider(visible=False)
          ),
          yaxis=go.layout.YAxis(
              side="right",
              title='Price',
              range=[low, high],
          ),
          xaxis2=go.layout.XAxis(
              side="top",
              showgrid=False,
              ticks='',
              showticklabels=False,
              range=[0, 2 * max(volprofile.values)],
              overlaying="x",
          ),
          yaxis2=go.layout.YAxis(
              side="left",
              range=[low, high],
              showticklabels=False,
              overlaying="y2",
          ),
      )

      fig = go.Figure(data=[fig1, fig2], layout=layout)

      fig.update_layout(autosize=False, width=1280, height=720, title_text=str(symbol.upper()) + 'USDT 5min',
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, b=10, t=50),
                        font=dict(size=10, color="#e1e1e1"),
                        paper_bgcolor="#1e1e1e",
                        plot_bgcolor="#1e1e1e",
                        legend=dict(orientation="h"))

      fig.add_hline(y=vah, annotation_text='VAH ' + vah_text, annotation_position="top left", line_color='yellow',
                    line_dash="dash")
      fig.add_hline(y=val, annotation_text='VAL ' + val_text, annotation_position="bottom left",
                    line_color='yellow',
                    line_dash="dash")
      fig.add_hline(y=poc, line_color="red", annotation_text='POC ' + poc_text, annotation_position="top left")
      fig.add_trace(go.Scatter(x=[today.index[0], today.index[0]],
                               y=[min(current_chart['Low']), max(current_chart['High'])], mode='lines',
                               line=dict(color='white', width=1, dash='dot'),
                               name='New Day'))

      fig.layout.yaxis.showgrid = False
      fig.layout.yaxis2.showgrid = False
      fig.layout.xaxis.showgrid = False
      fig.layout.xaxis2.showgrid = False

      config = {'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'], 'displaylogo': False, 'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toImage', 'resetScale',  'zoomIn', 'zoomOut']}

      st.plotly_chart(fig, use_container_width=True, config=config)
      
    except Exception as e:
      st.error('Sorry. There was an error.')

if chart == 'Volume Profile' and timeframe == 'Weekly':
  
    st.markdown(f"<h3 style='text-align: left;'>Previous Week's {chart} of {symbol}</h3>",
                unsafe_allow_html=True)
    
    try:

      df = pd.DataFrame.from_dict(db)

      df['Date'] = pd.to_datetime(df['Date'])

      weekly_vp = df.copy()

      weekly_vp['Volume'] = weekly_vp['xAsks'] + weekly_vp['xBids']

      binancetime = datetime.utcfromtimestamp(client.get_server_time()['serverTime'] / 1000)

      day_of_week = int(binancetime.isoweekday() - 1 + 7)

      prev_week_end = binancetime.replace(hour=0, minute=0, second=0, microsecond=0)
      prev_week_start = binancetime.replace(hour=0, minute=0, second=0, microsecond=0)

      prev_week_start = prev_week_start - timedelta(days=day_of_week)

      prev_week_end = prev_week_start + timedelta(days=7)
      prev_week_end = prev_week_end - timedelta(minutes=5)

      weekly_vp['Date'] = pd.to_datetime(weekly_vp['Date'])

      weekly_vp.set_index('Date', inplace=True)

      weekly_vp = weekly_vp.loc[prev_week_start:prev_week_end]

      this_week = df.copy()

      this_week['Date'] = pd.to_datetime(this_week['Date'])

      week_start = prev_week_start + timedelta(days=7)

      this_week.set_index('Date', inplace=True)

      this_week = this_week.loc[week_start:]

      # Volume Profile 1Week
      bucket_size = 0.002 * max(weekly_vp['Close'])
      volprofile = weekly_vp['Volume'].groupby(
          weekly_vp['Close'].apply(lambda x: bucket_size * round(x / bucket_size, 0))).sum()
      VPOC = volprofile.max()

      poc = volprofile.idxmax()
      volprofile_copy = volprofile.copy()
      total_volume = volprofile_copy.sum()
      value_area_volume = total_volume * 0.70

      val = 0
      vah = 0
      while value_area_volume >= 0:
          for x in range(len(volprofile_copy)):
              if volprofile_copy.values[x] == VPOC:
                  two_above = volprofile.values[x - 1] + volprofile_copy.values[x - 2] if x - 2 >= 0 else 0
                  two_below = volprofile.values[x + 1] + volprofile_copy.values[x + 2] if x + 2 <= len(
                      volprofile_copy) - 1 else 0
                  val = volprofile_copy.index[x - 1] if x - 1 >= 0 else val
                  vah = volprofile_copy.index[x + 1] if x + 1 <= len(volprofile_copy) - 1 else vah
                  if two_above >= two_below:
                      volprofile_copy.drop([volprofile_copy.index[x - 1], volprofile_copy.index[x - 2]],
                                           inplace=True)
                      value_area_volume = value_area_volume - two_above
                      break
                  else:
                      volprofile_copy.drop([volprofile_copy.index[x + 1], volprofile_copy.index[x + 2]],
                                           inplace=True)
                      value_area_volume = value_area_volume - two_below
                      break

      vah_value = round(vah, 3)
      val_value = round(val, 3)
      poc_value = round(poc, 3)

      vah_text = '{0:,}'.format(vah_value)
      val_text = '{0:,}'.format(val_value)
      poc_text = '{0:,}'.format(poc_value)

      vah_text = str(vah_text)
      val_text = str(val_text)
      poc_text = str(poc_text)

      current_chart = pd.concat([weekly_vp, this_week], axis=0)

      agg_dict = {'Open': 'first',
                  'High': 'max',
                  'Low': 'min',
                  'Close': 'last',
                  'xBids': 'sum',
                  'xAsks': 'sum', }

      current_chart = current_chart.resample(rule='H').agg(agg_dict)

      fig1 = go.Candlestick(
          x=current_chart.index,
          open=current_chart.Open,
          high=current_chart.High,
          low=current_chart.Low,
          close=current_chart.Close,
          xaxis='x',
          yaxis='y',
          visible=True,
          showlegend=False,
          increasing_fillcolor='#24A06B',
          decreasing_fillcolor="#CC2E3C",
          increasing_line_color='#2EC886',
          decreasing_line_color='#FF3A4C',
          line=dict(width=1), opacity=1)

      fig2 = go.Bar(
          x=volprofile.values,
          y=volprofile.keys().values,
          orientation='h',
          xaxis='x2',
          yaxis='y',
          visible=True,
          showlegend=False,
          name='Volume Bars',
          marker_color='dodgerblue',
          opacity=0.2,
          text=volprofile.values,
          textposition='auto'
      )

      low = min(current_chart['Low'])
      high = max(current_chart['High'])
      layout = go.Layout(
          title=go.layout.Title(text="Volume Profile"),
          xaxis=go.layout.XAxis(
              side="bottom",
              title="Date",
              rangeslider=go.layout.xaxis.Rangeslider(visible=False)
          ),
          yaxis=go.layout.YAxis(
              side="right",
              title='Price',
              range=[low, high],
          ),
          xaxis2=go.layout.XAxis(
              side="top",
              showgrid=False,
              ticks='',
              showticklabels=False,
              range=[0, 2 * max(volprofile.values)],
              overlaying="x",
          ),
          yaxis2=go.layout.YAxis(
              side="left",
              range=[low, high],
              showticklabels=False,
              overlaying="y2",
          ),
      )

      fig = go.Figure(data=[fig1, fig2], layout=layout)

      fig.update_layout(autosize=False, width=1280, height=720, title_text=str(symbol.upper()) + 'USDT 1hr',
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, b=10, t=50),
                        font=dict(size=10, color="#e1e1e1"),
                        paper_bgcolor="#1e1e1e",
                        plot_bgcolor="#1e1e1e",
                        legend=dict(orientation="h"))

      fig.add_hline(y=vah, annotation_text='VAH ' + vah_text, annotation_position="top left", line_color='yellow',
                    line_dash="dash")
      fig.add_hline(y=val, annotation_text='VAL ' + val_text, annotation_position="bottom left",
                    line_color='yellow',
                    line_dash="dash")
      fig.add_hline(y=poc, line_color="red", annotation_text='POC ' + poc_text, annotation_position="top left")
      fig.add_trace(go.Scatter(x=[this_week.index[0], this_week.index[0]],
                               y=[min(current_chart['Low']), max(current_chart['High'])], mode='lines',
                               line=dict(color='white', width=1, dash='dot'),
                               name='New Week'))

      fig.layout.yaxis.showgrid = False
      fig.layout.yaxis2.showgrid = False
      fig.layout.xaxis.showgrid = False
      fig.layout.xaxis2.showgrid = False

      config = {'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'], 'displaylogo': False, 'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toImage', 'resetScale',  'zoomIn', 'zoomOut']}

      st.plotly_chart(fig, use_container_width=True, config=config)
    
    except Exception as e:
      st.error('Sorry. There was an error.')

    
if chart == 'Order Flow' and timeframe == '5min':
  
    try:
    
      binancetime = datetime.utcfromtimestamp(client.get_server_time()['serverTime'] / 1000)
      
      isNewDay = binancetime.hour == 0 and binancetime.minute < 20

      df3 = pd.DataFrame.from_dict(db)

      df3['Volume'] = df3['xAsks'] + df3['xBids']
      df3['Volume_Avg'] = df3['Volume'].rolling(12).mean()

      df3['Buy_Volume'] = ' '
      df3['Sell_Volume'] = ' '

      for x in df3.index:
          if df3.loc[x, 'xAsks'] > df3.loc[x, 'xBids']:
              df3.loc[x, 'Buy_Volume'] = abs(df3.loc[x, 'Volume'])
          if df3.loc[x, 'xBids'] > df3.loc[x, 'xAsks']:
              df3.loc[x, 'Sell_Volume'] = abs(df3.loc[x, 'Volume'])

      # OFR

      df3['OFR'] = (df3['xAsks'] - df3['xBids']) / (df3['xAsks'] + df3['xBids']) * 100

      df3['OFR_Avg'] = df3['OFR'].rolling(36).mean()

      df3['Bull'] = ' '
      df3['Bear'] = ' '

      for x in df3.index:
          if df3.loc[x, 'OFR_Avg'] < 0:
              df3.loc[x, 'Bear'] = abs(df3.loc[x, 'OFR_Avg'])
          if df3.loc[x, 'OFR_Avg'] >= 0:
              df3.loc[x, 'Bull'] = abs(df3.loc[x, 'OFR_Avg'])

      # Slice Dataframe To Only Show Past 24hrs
      m5 = df3.iloc[-288:, :]

      if isNewDay:
        now = now - timedelta(minutes=30)
        now = now.strftime("%Y-%m-%d %H:%M:%S")
      else:
        now = now.strftime("%Y-%m-%d %H:%M:%S")

      m5 = m5.reset_index()

      m5.set_index('Date', inplace=True)

      for idx, row in m5.loc[now:].iterrows():
          m5.loc[idx, 'Delta'] = m5.loc[idx, 'xAsks'] - m5.loc[idx, 'xBids']

      m5['CVD'] = m5['Delta'].cumsum()

      vn = m5.loc[now:]

      vn['Volume'] = vn['xAsks'] + vn['xBids']

      bucket_size = 0.002 * max(vn['Close'])
      volprofile = vn['Volume'].groupby(
          vn['Close'].apply(lambda x: bucket_size * round(x / bucket_size, 0))).sum()
      VPOC = volprofile.max()
      if len(volprofile) >= 3:
          volume_nodes = volprofile.nlargest(n=3).keys().tolist()

          poc = volprofile.idxmax()
          volume_nodes.remove(poc)

          vol_node1 = volprofile[volume_nodes[0]]
          vol_node2 = volprofile[volume_nodes[1]]
          max_node = volprofile[poc]

          vol_node1 = int(vol_node1)
          vol_node2 = int(vol_node2)
          max_node = int(max_node)

          level1 = volume_nodes[0]
          level2 = volume_nodes[1]
          level3 = poc

          level1 = round(level1, 3)
          level2 = round(level2, 3)
          level3 = round(level3, 3)

          vol_node1 = '{0:,}'.format(vol_node1)
          vol_node2 = '{0:,}'.format(vol_node2)
          max_node = '{0:,}'.format(max_node)

          level1 = '{0:,}'.format(level1)
          level2 = '{0:,}'.format(level2)
          level3 = '{0:,}'.format(level3)

      # Plot 5min Chart On a 24hr Rolling Basis
      fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                          vertical_spacing=0.03, subplot_titles=('Price', 'CVD', 'OF', 'Volume'),
                          row_width=[0.2, 0.2, 0.4, 0.7])

      fig.add_trace(
          go.Candlestick(
              x=m5.index,
              open=m5['Open'],
              high=m5['High'],
              low=m5['Low'],
              close=m5['Close'],
              name='Price',
              increasing_fillcolor='#24A06B',
              decreasing_fillcolor="#CC2E3C",
              increasing_line_color='#2EC886',
              decreasing_line_color='#FF3A4C',
              line=dict(width=1), opacity=1), row=1, col=1)

      fig.add_trace(
          go.Scatter(x=m5.index, y=m5.CVD, name='Cumulative Delta'), row=2, col=1)

      fig.add_trace(
          go.Bar(x=m5.index, y=m5.Bull, marker_color='green', marker_line_color='green', opacity=1,
                 name='Bullish OF'), row=3, col=1)

      fig.add_trace(
          go.Bar(x=m5.index, y=m5.Bear, marker_color='red', marker_line_color='red', opacity=1,
                 name='Bearish OF'), row=3, col=1)

      fig.add_trace(
          go.Bar(x=m5.index, y=m5.Buy_Volume, marker_color='green', marker_line_color='green', opacity=1,
                 name='Bullish Volume'), row=4, col=1)

      fig.add_trace(
          go.Bar(x=m5.index, y=m5.Sell_Volume, marker_color='red', marker_line_color='red', opacity=1,
                 name='Bearish Volume'), row=4, col=1)

      fig.add_trace(
          go.Scatter(x=m5.index, y=m5.Volume_Avg, name='Volume Avg', marker_color="yellow", line_width=1), row=4, col=1)

      if len(volprofile) >= 3:
          annotation1 = str(vol_node1) + ' at $' + str(level1)
          annotation2 = str(vol_node2) + ' at $' + str(level2)
          annotation3 = str(max_node) + ' at $' + str(level3)

          fig.add_hline(y=volume_nodes[0], annotation_text=annotation1, row=1, annotation_position="top left")
          fig.add_hline(y=volume_nodes[1], annotation_text=annotation2, row=1, annotation_position="top left")
          fig.add_hline(y=poc, line_color="red", annotation_text=annotation3, row=1, annotation_position="top left")

      fig.update_layout(autosize=False, width=1280, height=720, title_text=symbol.upper() + 'USDT 5min',
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, b=10, t=50),
                        font=dict(size=10, color="#e1e1e1"),
                        paper_bgcolor="#1e1e1e",
                        plot_bgcolor="#1e1e1e",
                        legend=dict(orientation="h"))
      fig.update_xaxes(
          gridcolor="#1f292f",
          showgrid=True, )

      fig.layout.yaxis.showgrid = False
      fig.layout.yaxis2.showgrid = False
      fig.layout.xaxis.showgrid = False

      config = {'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'], 'displaylogo': False, 'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toImage', 'resetScale',  'zoomIn', 'zoomOut']}

      st.plotly_chart(fig, use_container_width=True, config=config)
      
    except Exception as e:
      st.error('Sorry. There was an error.')

if chart == 'Order Flow' and timeframe == '1hr':
  
    try:

      binancetime = datetime.utcfromtimestamp(client.get_server_time()['serverTime'] / 1000)

      df3 = pd.DataFrame.from_dict(db)

      bars = int(int(binancetime.isoweekday()) * 288) + int((binancetime.hour * 12)) + int((binancetime.minute / 5))

      week_day = binancetime.isoweekday() -1

      start = binancetime.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=week_day)

      start = start.strftime("%Y-%m-%d %H:%M:%S")

      h1 = df3

      h1['Date'] = pd.to_datetime(h1['Date'])

      h1.set_index('Date', inplace=True)

      h1 = h1.loc[start:]

      agg_dict = {'Open': 'first',
                  'High': 'max',
                  'Low': 'min',
                  'Close': 'last',
                  'xBids': 'sum',
                  'xAsks': 'sum', }

      h1 = h1.resample(rule='H').agg(agg_dict)

      h1['Volume'] = h1['xAsks'] + h1['xBids']
      h1['Volume_Avg'] = h1['Volume'].rolling(12).mean()

      h1['Buy_Volume'] = ' '
      h1['Sell_Volume'] = ' '

      for x in h1.index:
          if h1.loc[x, 'xAsks'] > h1.loc[x, 'xBids']:
              h1.loc[x, 'Buy_Volume'] = abs(h1.loc[x, 'Volume'])
          if h1.loc[x, 'xBids'] > h1.loc[x, 'xAsks']:
              h1.loc[x, 'Sell_Volume'] = abs(h1.loc[x, 'Volume'])

      # OFR

      h1['OFR'] = (h1['xAsks'] - h1['xBids']) / (h1['xAsks'] + h1['xBids']) * 100

      h1['OFR_Avg'] = h1['OFR'].rolling(12).mean()

      h1['Bull'] = ' '
      h1['Bear'] = ' '

      for x in h1.index:
          if h1.loc[x, 'OFR_Avg'] < 0:
              h1.loc[x, 'Bear'] = abs(h1.loc[x, 'OFR_Avg'])
          if h1.loc[x, 'OFR_Avg'] >= 0:
              h1.loc[x, 'Bull'] = abs(h1.loc[x, 'OFR_Avg'])

      h1['Delta'] = h1['xAsks'] - h1['xBids']

      h1['CVD'] = h1['Delta'].cumsum()

      h1['Volume'] = h1['xAsks'] + h1['xBids']

      bucket_size = 0.002 * max(h1['Close'])
      volprofile = h1['Volume'].groupby(
          h1['Close'].apply(lambda x: bucket_size * round(x / bucket_size, 0))).sum()
      VPOC = volprofile.max()
      if len(volprofile) >= 3:
            volume_nodes = volprofile.nlargest(n=3).keys().tolist()

            poc = volprofile.idxmax()
            volume_nodes.remove(poc)

            vol_node1 = volprofile[volume_nodes[0]]
            vol_node2 = volprofile[volume_nodes[1]]
            max_node = volprofile[poc]

            vol_node1 = int(vol_node1)
            vol_node2 = int(vol_node2)
            max_node = int(max_node)

            level1 = volume_nodes[0]
            level2 = volume_nodes[1]
            level3 = poc

            level1 = round(level1, 3)
            level2 = round(level2, 3)
            level3 = round(level3, 3)

            vol_node1 = '{0:,}'.format(vol_node1)
            vol_node2 = '{0:,}'.format(vol_node2)
            max_node = '{0:,}'.format(max_node)

            level1 = '{0:,}'.format(level1)
            level2 = '{0:,}'.format(level2)
            level3 = '{0:,}'.format(level3)

      # Plot 1hr Chart On a 7Day Rolling Basis
      fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                          vertical_spacing=0.03, subplot_titles=('Price', 'CVD', 'OF', 'Volume'),
                          row_width=[0.2, 0.2, 0.4, 0.7])

      fig.add_trace(
          go.Candlestick(
              x=h1.index,
              open=h1['Open'],
              high=h1['High'],
              low=h1['Low'],
              close=h1['Close'],
              name='Price',
              increasing_fillcolor='#24A06B',
              decreasing_fillcolor="#CC2E3C",
              increasing_line_color='#2EC886',
              decreasing_line_color='#FF3A4C',
              line=dict(width=1), opacity=1), row=1, col=1)

      fig.add_trace(
          go.Scatter(x=h1.index, y=h1.CVD, name='Cumulative Delta'), row=2, col=1)

      fig.add_trace(
          go.Bar(x=h1.index, y=h1.Bull, marker_color='green', marker_line_color='green', opacity=1,
                 name='Bullish OF'), row=3, col=1)

      fig.add_trace(
          go.Bar(x=h1.index, y=h1.Bear, marker_color='red', marker_line_color='red', opacity=1,
                 name='Bearish OF'), row=3, col=1)

      fig.add_trace(
          go.Bar(x=h1.index, y=h1.Buy_Volume, marker_color='green', marker_line_color='green', opacity=1,
                 name='Bullish Volume'), row=4, col=1)

      fig.add_trace(
          go.Bar(x=h1.index, y=h1.Sell_Volume, marker_color='red', marker_line_color='red', opacity=1,
                 name='Bearish Volume'), row=4, col=1)

      fig.add_trace(
          go.Scatter(x=h1.index, y=h1.Volume_Avg, name='Volume Avg', marker_color="yellow", line_width=1), row=4, col=1)

      if len(volprofile) >= 3:
          annotation1 = str(vol_node1) + ' at $' + str(level1)
          annotation2 = str(vol_node2) + ' at $' + str(level2)
          annotation3 = str(max_node) + ' at $' + str(level3)

          fig.add_hline(y=volume_nodes[0], annotation_text=annotation1, row=1, annotation_position="top left")
          fig.add_hline(y=volume_nodes[1], annotation_text=annotation2, row=1, annotation_position="top left")
          fig.add_hline(y=poc, line_color="red", annotation_text=annotation3, row=1, annotation_position="top left")

      fig.update_layout(autosize=False, width=1280, height=720, title_text=symbol.upper() + 'USDT 1hr',
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, b=10, t=50),
                        font=dict(size=10, color="#e1e1e1"),
                        paper_bgcolor="#1e1e1e",
                        plot_bgcolor="#1e1e1e",
                        legend=dict(orientation="h"))
      fig.update_xaxes(
          gridcolor="#1f292f",
          showgrid=True, )

      fig.layout.yaxis.showgrid = False
      fig.layout.yaxis2.showgrid = False
      fig.layout.xaxis.showgrid = False

      config = {'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'], 'displaylogo': False, 'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toImage', 'resetScale',  'zoomIn', 'zoomOut']}

      st.plotly_chart(fig, use_container_width=True, config=config)
      
    except Exception as e:
      st.error('Sorry. There was an error.')

    
if chart == 'Order Flow' and timeframe == '15min':
  
    try:

      binancetime = datetime.utcfromtimestamp(client.get_server_time()['serverTime'] / 1000)

      df3 = pd.DataFrame.from_dict(db)

      start = int(864) + int((binancetime.hour * 12)) + int((binancetime.minute / 5))

      m15 = df3.iloc[-start:, :]

      m15['Date'] = pd.to_datetime(m15['Date'])

      m15.set_index('Date', inplace=True)

      agg_dict = {'Open': 'first',
                  'High': 'max',
                  'Low': 'min',
                  'Close': 'last',
                  'xBids': 'sum',
                  'xAsks': 'sum', }

      m15 = m15.resample(rule='15min').agg(agg_dict)

      m15['Volume'] = m15['xAsks'] + m15['xBids']
      m15['Volume_Avg'] = m15['Volume'].rolling(4).mean()

      m15['Buy_Volume'] = ' '
      m15['Sell_Volume'] = ' '

      for x in m15.index:
          if m15.loc[x, 'xAsks'] > m15.loc[x, 'xBids']:
              m15.loc[x, 'Buy_Volume'] = abs(m15.loc[x, 'Volume'])
          if m15.loc[x, 'xBids'] > m15.loc[x, 'xAsks']:
              m15.loc[x, 'Sell_Volume'] = abs(m15.loc[x, 'Volume'])

      # OFR

      m15['OFR'] = (m15['xAsks'] - m15['xBids']) / (m15['xAsks'] + m15['xBids']) * 100

      m15['OFR_Avg'] = m15['OFR'].rolling(9).mean()

      m15['Bull'] = ' '
      m15['Bear'] = ' '

      for x in m15.index:
          if m15.loc[x, 'OFR_Avg'] < 0:
              m15.loc[x, 'Bear'] = abs(m15.loc[x, 'OFR_Avg'])
          if m15.loc[x, 'OFR_Avg'] >= 0:
              m15.loc[x, 'Bull'] = abs(m15.loc[x, 'OFR_Avg'])

      m15['Delta'] = m15['xAsks'] - m15['xBids']

      m15['CVD'] = m15['Delta'].cumsum()

      m15['Volume'] = m15['xAsks'] + m15['xBids']

      bucket_size = 0.002 * max(m15['Close'])
      volprofile = m15['Volume'].groupby(
          m15['Close'].apply(lambda x: bucket_size * round(x / bucket_size, 0))).sum()
      VPOC = volprofile.max()

      if len(volprofile) >= 3:
          volume_nodes = volprofile.nlargest(n=3).keys().tolist()

          poc = volprofile.idxmax()
          volume_nodes.remove(poc)

          vol_node1 = volprofile[volume_nodes[0]]
          vol_node2 = volprofile[volume_nodes[1]]
          max_node = volprofile[poc]

          vol_node1 = int(vol_node1)
          vol_node2 = int(vol_node2)
          max_node = int(max_node)

          level1 = volume_nodes[0]
          level2 = volume_nodes[1]
          level3 = poc

          level1 = round(level1, 3)
          level2 = round(level2, 3)
          level3 = round(level3, 3)

          vol_node1 = '{0:,}'.format(vol_node1)
          vol_node2 = '{0:,}'.format(vol_node2)
          max_node = '{0:,}'.format(max_node)

          level1 = '{0:,}'.format(level1)
          level2 = '{0:,}'.format(level2)
          level3 = '{0:,}'.format(level3)

      # Plot 1hr Chart On a 7Day Rolling Basis
      fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                          vertical_spacing=0.03, subplot_titles=('Price', 'CVD', 'OF', 'Volume'),
                          row_width=[0.2, 0.2, 0.4, 0.7])

      fig.add_trace(
          go.Candlestick(
              x=m15.index,
              open=m15['Open'],
              high=m15['High'],
              low=m15['Low'],
              close=m15['Close'],
              name='Price',
              increasing_fillcolor='#24A06B',
              decreasing_fillcolor="#CC2E3C",
              increasing_line_color='#2EC886',
              decreasing_line_color='#FF3A4C',
              line=dict(width=1), opacity=1), row=1, col=1)

      fig.add_trace(
          go.Scatter(x=m15.index, y=m15.CVD, name='Cumulative Delta'), row=2, col=1)

      fig.add_trace(
          go.Bar(x=m15.index, y=m15.Bull, marker_color='green', marker_line_color='green', opacity=1,
                 name='Bullish OF'), row=3, col=1)

      fig.add_trace(
          go.Bar(x=m15.index, y=m15.Bear, marker_color='red', marker_line_color='red', opacity=1,
                 name='Bearish OF'), row=3, col=1)

      fig.add_trace(
          go.Bar(x=m15.index, y=m15.Buy_Volume, marker_color='green', marker_line_color='green', opacity=1,
                 name='Bullish Volume'), row=4, col=1)

      fig.add_trace(
          go.Bar(x=m15.index, y=m15.Sell_Volume, marker_color='red', marker_line_color='red', opacity=1,
                 name='Bearish Volume'), row=4, col=1)

      fig.add_trace(
          go.Scatter(x=m15.index, y=m15.Volume_Avg, name='Volume Avg', marker_color="yellow", line_width=1), row=4, col=1)

      if len(volprofile) >= 3:
          annotation1 = str(vol_node1) + ' at $' + str(level1)
          annotation2 = str(vol_node2) + ' at $' + str(level2)
          annotation3 = str(max_node) + ' at $' + str(level3)

          fig.add_hline(y=volume_nodes[0], annotation_text=annotation1, row=1, annotation_position="top left")
          fig.add_hline(y=volume_nodes[1], annotation_text=annotation2, row=1, annotation_position="top left")
          fig.add_hline(y=poc, line_color="red", annotation_text=annotation3, row=1, annotation_position="top left")

      fig.update_layout(autosize=False, width=1280, height=720, title_text=symbol.upper() + 'USDT 15min',
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=10, r=10, b=10, t=50),
                        font=dict(size=10, color="#e1e1e1"),
                        paper_bgcolor="#1e1e1e",
                        plot_bgcolor="#1e1e1e",
                        legend=dict(orientation="h"))
      fig.update_xaxes(
          gridcolor="#1f292f",
          showgrid=True, )

      fig.layout.yaxis.showgrid = False
      fig.layout.yaxis2.showgrid = False
      fig.layout.xaxis.showgrid = False

      config = {'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'], 'displaylogo': False, 'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'toImage', 'resetScale',  'zoomIn', 'zoomOut']}

      st.plotly_chart(fig, use_container_width=True, config=config)
      
    except Exception as e:
      st.error('Sorry. There was an error.')
