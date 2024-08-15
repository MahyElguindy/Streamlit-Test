import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
from datetime import date
import os
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pickle
from pathlib import Path
import pygsheets
import datetime
from streamlit_js_eval import *
from streamlit.components.v1 import html


st.set_page_config(page_title="TBS-Dashboard", page_icon="🍩", layout="wide")
screen_width = streamlit_js_eval(js_expressions='screen.width', key = 'SCR')
# screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH',  want_output = True,)  

# --- USER AUTHENTICATION ---
names = ["Info Traff", "TBS"]
usernames = ["InfoTraff", "TBS"]


with open('./login.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login(location="main", fields={"Login":"Login"})

if authentication_status:
    # config = {'displayModeBar': True, 'staticPlot': True}
    config = {'displayModeBar': False}
    text_size = 25
    
    # .stPlotlyChart {{
    #     outline: 10px solid {PLOT_BGCOLOR};
    #     border-radius: 5px;
    #     box-shadow: rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px;
    #     }}

    # st.write(screen_width)
    
    ############CSS IMPORT################
    with open('./css/style.css') as f:
        css = f.read()

    st.markdown(
        f"""
        <style>
        {css}
        </style>
        """, unsafe_allow_html=True
    )
    
    new_title = '<h1>Welcome to your TBS Dashboard 🍩</h1> <br>'
    st.markdown(new_title, unsafe_allow_html=True)

    # dashboard variables
    gc_dashboard = pygsheets.authorize(service_file='./shoptalk.json')
    sh_dashboard = gc_dashboard.open('TBS-entry')  # Open GoogleSheet
    ws_dashboard = sh_dashboard.worksheet('title', 'Sheet1')
    df = ws_dashboard.get_as_df()

    sh_warnings = gc_dashboard.open('TBS-warning')  # Open GoogleSheet
    ws_warnings = sh_warnings.worksheet('title', 'Sheet1')
    df_warnings = ws_warnings.get_as_df()

    
    df['Time'] = pd.to_datetime(df['Time'])
    df_warnings['Time'] = pd.to_datetime(df_warnings['Time'])

        # Conditional assignment
    df_warnings['Count'] = df_warnings.apply(lambda row: 1 if row['Type'] == 'transaction_num' and \
                                                                   row['Count'] > 1 else 0 if row['Type'] == 'transaction_num' and \
                                                                    row['Count'] == 1 else row['Count'], axis=1)
    
    day = st.date_input('Select day', min_value=min(df['Time']), max_value=max(df['Time']), value= df['Time'].iloc[-1])
    
    # st.markdown(f'''<p class= "date">Date {day}</p>''', unsafe_allow_html=True)

    previous_day = day - datetime.timedelta(days=1)
    

    # Filter the DataFrame based on the selected day
    filtered_df = df[df['Time'].dt.date == day]
    filtered_df_warnings = df_warnings[df_warnings['Time'].dt.date == day]
    filtered_df_warnings_previous = df_warnings[df_warnings['Time'].dt.date == previous_day]

    # Group by hour and class on the filtered DataFrame
    warnings_count = filtered_df_warnings.groupby(['Type'])['Count'].sum().reset_index()
    warnings_count_previous = filtered_df_warnings_previous.groupby(['Type'])['Count'].sum().reset_index()
    
    custom_order = ['spoilageCount', 'spoilage', 'pastry low', 'boxes low',
                    'salads low', 'missed opportunity', 'overstaffed',
                    'understaffed', 'long queue', 'hat missing food station',
                    'hat missing kitchen', 'transaction', 'transaction_num']
    warnings_count['Type'] = pd.Categorical(warnings_count['Type'], categories=custom_order, ordered=True)
    warnings_count = warnings_count.sort_values(by='Type', ignore_index = True)

    warnings_count_previous['Type'] = pd.Categorical(warnings_count_previous['Type'], categories=custom_order, ordered=True)
    warnings_count_previous = warnings_count_previous.sort_values(by='Type', ignore_index = True)

    # cards logic
    if len(warnings_count):
        dict_cards = {'total':'', 'spoilage':'', 'missed opportunity': '',
                      'overstaffed': '', 'understaffed': '', 'long queue': '',
                      'hat missing food station': '', 'hat missing kitchen': '',
                      'pastry low': '', 'boxes low': '', 'salads low': ''}
        warnings_cards = ''
        total_loss = 0
        total_count = 0
        total_loss_previous = 0
        diff_loss = 0
        for i in range(len(warnings_count_previous)):
            if warnings_count_previous.loc[i, "Type"] == 'pastry low':
                total_loss_previous += warnings_count_previous.loc[i, "Count"]
            elif warnings_count_previous.loc[i, "Type"] == 'boxes low':
                total_loss_previous += warnings_count_previous.loc[i, "Count"]
            elif warnings_count_previous.loc[i, "Type"] == 'salads low':
                total_loss_previous += warnings_count_previous.loc[i, "Count"]

        for i in range(len(warnings_count)):
            if warnings_count.loc[i, "Type"] == 'pastry low':
                total_loss += warnings_count.loc[i, "Count"]
                total_count += warnings_count.loc[i, "Count"]
            elif warnings_count.loc[i, "Type"] == 'boxes low':
                total_loss += warnings_count.loc[i, "Count"]
                total_count += warnings_count.loc[i, "Count"]
            elif warnings_count.loc[i, "Type"] == 'salads low':
                total_loss += warnings_count.loc[i, "Count"]
                total_count += warnings_count.loc[i, "Count"]
            

        if total_loss > total_loss_previous:
            diff_loss = f'+{abs(total_loss - total_loss_previous)}'
            cost_color = 'red'
        elif total_loss < total_loss_previous:
            diff_loss = f'-{abs(total_loss - total_loss_previous)}'
            cost_color = 'green'
        else:
            diff_loss = '0'
            cost_color = 'black'
        
        if abs(total_loss - total_loss_previous) == 0:
            warnings_cards += f'''<div class="warning" style="text-align:center;">
                        <p class="first">Total Low Stock Alerts</p>
                        <p class="second">Count: {total_loss}</p>
                        <p class="third" style="color:{cost_color};"><span style="color:#999999;">Same as day before</span></p>
                        </div>'''
            dict_cards['total'] = f'''
                            <div class="warning-parent">
                            <div class="warning" style="text-align:center;">
                            <p class="first">Total Low Stock Alerts</p>
                            <p class="second">Count: {total_loss}</p>
                            <p class="third" style="color:{cost_color};"><span style="color:#999999;">Same as day before</span></p>
                            </div>
                        </div>'''
        else:
            warnings_cards += f'''<div class="warning" style="text-align:center;">
                            <p class="first">Total Low Stock Alerts</p>
                            <p class="second">Count: {total_loss}</p>
                            <p class="third" style="color:{cost_color};">{diff_loss} <span style="color:#999999;">from day before</span></p>
                            </div>'''
            dict_cards['total'] = f'''
                            <div class="warning-parent">
                            <div class="warning" style="text-align:center;">
                            <p class="first">Total Low Stock Alerts</p>
                            <p class="second">Count: {total_loss}</p>
                            <p class="third" style="color:{cost_color};">{diff_loss} <span style="color:#999999;">from day before</span></p>
                            </div>
                            </div>'''
        # st.markdown(, unsafe_allow_html=True)
        
        for i in range(len(warnings_count)):
            spoilage_flag = True
            type_name = warnings_count.loc[i, "Type"].title()
            type_count = warnings_count.loc[i, "Count"]
            type_cost = ''
            cost_color = 'black'
            count_types = ['missed opportunity', 'overstaffed', 'understaffed',
                        'long queue', 'hat missing food station', 'hat missing kitchen', 'pastry low', 'boxes low', 'salads low']
            if warnings_count.loc[i, "Type"] == 'missed opportunity':
                type_name = 'Unattended Client'
            elif warnings_count.loc[i, "Type"] == 'spoilage':
                continue
            elif warnings_count.loc[i, "Type"] == 'transaction':
                continue
            elif warnings_count.loc[i, "Type"] == 'transaction_num':
                continue
            elif warnings_count.loc[i, "Type"] == 'spoilageCount':
                type_name = 'Spoilage'
                try:
                    previous_count = warnings_count_previous.loc[warnings_count_previous['Type'] == warnings_count.loc[i, "Type"],
                                                                'Count'].sum()
        
                    if type_count > previous_count:
                        type_cost = f'+{abs(type_count - previous_count)}'
                        cost_color = 'red'
                    elif type_count < previous_count:
                        type_cost = f'-{abs(type_count - previous_count)}'
                        cost_color = 'green'
                    else:
                        type_cost = f'0'
                        cost_color = 'black'

                    # if previous_count == 0:
                    #     type_cost = f'0'
                    #     cost_color = 'black'
                    
                    if type_count == 0 and previous_count == 0:
                        spoilage_flag = False

                except:
                    type_cost = f'0'
                    pass
                
                if abs(type_count - previous_count) == 0:
                    warnings_cards += f'''<div class="warning">
                            <p class="first">{type_name}</p>
                            <p class="second">Count: {warnings_count.loc[i, "Count"]}</p>
                            <p class="third" style="color:{cost_color};"><span style="color:#999999;">Same as day before</span></p>
                            </div>'''
                
                    dict_cards['spoilage'] = f'''
                            <div class="warning-parent">
                            <div class="warning">
                            <p class="first">{type_name}</p>
                            <p class="second">Count: {warnings_count.loc[i, "Count"]}</p>
                            <p class="third" style="color:{cost_color};"><span style="color:#999999;">Same as day before</span></p>
                            </div>
                            </div>'''
                    
                else:
                    warnings_cards += f'''<div class="warning">
                            <p class="first">{type_name}</p>
                            <p class="second">Count: {warnings_count.loc[i, "Count"]}</p>
                            <p class="third" style="color:{cost_color};">{type_cost} <span style="color:#999999;">from day before</span></p>
                            </div>'''
                    
                    dict_cards['spoilage'] = f'''
                            <div class="warning-parent">
                            <div class="warning">
                            <p class="first">{type_name}</p>
                            <p class="second">Count: {warnings_count.loc[i, "Count"]}</p>
                            <p class="third" style="color:{cost_color};">{type_cost} <span style="color:#999999;">from day before</span></p>
                            </div>
                            </div>'''
                continue
                
            if warnings_count.loc[i, "Type"] in count_types:
                try:
                    previous_count = warnings_count_previous.loc[warnings_count_previous['Type'] == warnings_count.loc[i, "Type"],
                                                                'Count'].sum()
                    
                    if type_count == previous_count or previous_count == 0:
                        type_cost = f''
                        cost_color = 'black'
                    elif type_count > previous_count:
                        type_cost = f'+{abs(type_count - previous_count)} <span style="color:#999999;">from day before</span>'
                        cost_color = 'red'
                    elif type_count < previous_count:
                        type_cost = f'-{abs(type_count - previous_count)} <span style="color:#999999;">from day before</span>'
                        cost_color = 'green'
                    # else:
                    #     type_cost = ''
                    #     cost_color = 'black'
                except:
                    pass
            else:
                cost_color = 'black'

            warnings_cards += f'''<div class="warning">
                        <p class="first">{type_name}</p>
                        <p class="second">Count: {type_count}</p>
                        <p class="third" style="color:{cost_color};">{type_cost}</p>
                        </div>'''
            
            dict_cards[warnings_count.loc[i, "Type"]] = f'''
                        <div class="warning-parent">
                        <div class="warning">
                        <p class="first">{type_name}</p>
                        <p class="second">Count: {type_count}</p>
                        <p class="third" style="color:{cost_color};">{type_cost}</p>
                        </div>
                        </div>'''
            
        try:
            if screen_width == None or screen_width <= 800:
                if dict_cards['spoilage'] != '':
                    st.markdown(dict_cards['spoilage'], unsafe_allow_html= True)
    
    
                if dict_cards['total'] == '':
                    st.markdown(f'''
                            <div class="warning-parent">
                            <div class="warning">
                            <p class="first">Total Low Stock Alerts</p>
                            <p class="second"></p>
                            <p class="third" style="color:{cost_color};"></p>
                            </div>
                            </div>''', unsafe_allow_html= True)
                else:
                    st.markdown(dict_cards['total'], unsafe_allow_html= True)
                
                with st.expander('see more'):
                    if dict_cards['pastry low'] != '':
                        st.markdown(dict_cards['pastry low'], unsafe_allow_html= True)
                    if dict_cards['salads low'] != '':
                        st.markdown(dict_cards['salads low'], unsafe_allow_html= True)
                    if dict_cards['boxes low'] != '':
                        st.markdown(dict_cards['boxes low'], unsafe_allow_html= True)
                    if dict_cards['missed opportunity'] != '':
                        st.markdown(dict_cards['missed opportunity'], unsafe_allow_html= True)
                    if dict_cards['overstaffed'] != '':
                        st.markdown(dict_cards['overstaffed'], unsafe_allow_html= True)
                    if dict_cards['understaffed'] != '':
                        st.markdown(dict_cards['understaffed'], unsafe_allow_html= True)
                    if dict_cards['long queue'] != '':
                        st.markdown(dict_cards['long queue'], unsafe_allow_html= True)
                    if dict_cards['hat missing food station'] != '':
                        st.markdown(dict_cards['hat missing food station'], unsafe_allow_html= True)
                    if dict_cards['hat missing kitchen'] != '':
                        st.markdown(dict_cards['hat missing kitchen'], unsafe_allow_html= True)
            
            elif screen_width > 800:
                st.markdown(f'''<div class="warning-parent">
                            {warnings_cards}
                            </div>''',
                            unsafe_allow_html= True)
        except:
            st.markdown(f'''<div class="warning-parent">
                            {warnings_cards}
                            </div>''',
                            unsafe_allow_html= True)

    # change column name
    filtered_df_warnings.rename(columns= {'Type': 'Class'}, inplace= True)

    # Group by hour and class on the filtered DataFrame
    enter_by_hour = filtered_df.groupby([filtered_df['Time'].dt.strftime('%I %p'),
                                         'Class'])['Count'].sum().reset_index()
    enter_by_hour = pd.concat([enter_by_hour,
                               filtered_df_warnings.groupby(
                                   [filtered_df_warnings['Time'].dt.strftime('%I %p'),
                                    'Class'])['Count'].sum().reset_index()])
    
    enter_by_class = df.groupby(['Class'])['Count'].sum().reset_index()

    # enter by hour for person
    person_per_hour = enter_by_hour.loc[(enter_by_hour['Class'] == 'Person') | 
                                        (enter_by_hour['Class'] == 'transaction') |
                                        (enter_by_hour['Class'] == 'transaction_num') |
                                        (enter_by_hour['Class'] == 'cluster') ]
    

    # enter by hour for black_employee
    white_employee_per_hour = enter_by_hour.loc[enter_by_hour['Class']
                                                == 'White_employee']
    
    # change classes names
    enter_by_hour['Class'] = enter_by_hour['Class'].replace({'White_employee': 'White Employee',
                                                             'Person': 'Customer'})
    person_per_hour.loc['Class'] = person_per_hour['Class'].replace({
        'Person':'Customer',
        'transaction':'Transaction',
        'transaction_num':'Groups Cash',
        'cluster': 'Groups Entry'}, inplace=True)
    
    white_employee_per_hour = white_employee_per_hour.replace(
        'White_employee', 'Employee')
    

    colors = {
        'Person': '#61BB6D',
        # 'Customer': '#61BB6D',
        'Customer': '#ffd633',
        'Employee': '#ffd633',
        # 'Transaction': '#ffd633',
        'Transaction': '#61BB6D',
        # 'Groups': '#ffd633'
        'Groups Cash': '#ff6600',
        'Groups Entry': '#e0e0d1'
    }

    # sorting dataframe
    person_per_hour[['x', 'y']] = person_per_hour['Time'].str.split(n=1, expand=True)
    person_per_hour = person_per_hour.sort_values(by= ['y', 'x'], ignore_index = True)

    temp = person_per_hour[(person_per_hour['x'] == '12') & (person_per_hour['y'] == 'PM')].iloc[:, :]
    temp_am = person_per_hour[(person_per_hour['x'] < '12') & (person_per_hour['y'] == 'AM')].iloc[:, :]
    temp_pm = person_per_hour[(person_per_hour['x'] < '12') & (person_per_hour['y'] == 'PM')].iloc[:, :]
    
    person_per_hour_sorted = pd.concat([temp_am, temp, temp_pm])

    # try:
    #     last_am_index = person_per_hour[person_per_hour['y'] == 'AM'].index[-1]
    #     index_12pm = person_per_hour[person_per_hour['x'] == '12'].index[-1] - 3
    #     person_per_hour_sorted = pd.concat([person_per_hour.iloc[:last_am_index + 1],
    #                                         person_per_hour.iloc[index_12pm: index_12pm + 4],
    #                                         person_per_hour.iloc[last_am_index + 1: index_12pm]])
    
    # except:
    #     try:
    #         last_am_index = person_per_hour[person_per_hour['y'] == 'AM'].index[-1]
    #         index_12pm = person_per_hour[person_per_hour['x'] == '12'].index[-1] - 2
    #         person_per_hour_sorted = pd.concat([person_per_hour.iloc[:last_am_index + 1],
    #                                             person_per_hour.iloc[index_12pm: index_12pm + 3],
    #                                             person_per_hour.iloc[last_am_index + 1: index_12pm]])
    #     except:
    #         try:
    #             last_am_index = person_per_hour[person_per_hour['y'] == 'AM'].index[-1]
    #             index_12pm = person_per_hour[person_per_hour['x'] == '12'].index[-1]
    #             person_per_hour_sorted = pd.concat([person_per_hour.iloc[:last_am_index + 1],
    #                                                 person_per_hour.iloc[index_12pm: index_12pm + 1],
    #                                                 person_per_hour.iloc[last_am_index + 1: index_12pm]])
    #         except:
    #             person_per_hour_sorted = person_per_hour.copy()
    
    enter_stat = person_per_hour_sorted[person_per_hour_sorted["Class"] == "Customer"]["Count"]
    st.markdown(
        f'<div class="container">' +
            '<p>TBS Entrances By Hour</p>' +
            '<div class="containerData">' +
            '<div class="boxData">' +
                '<p class="firstP">Enter Today</p>' +
                f'<p class="secondP">{int(enter_stat.sum())}</p>' +
            '</div>' +
            '<div class="boxData">' +
                '<p class="firstP">Enter Last Hour</p>' +
                f'<p class="secondP">{int(enter_stat.iloc[-1])}</p>' +
            '</div>' +
            '</div>'
        '</div>'
        , unsafe_allow_html=True
    )

    # new line
    st.markdown(f'<br>', unsafe_allow_html=True)
    
    fig = px.bar(person_per_hour_sorted, x="Time", y="Count", color="Class",
                color_discrete_map=colors, barmode='group')
    fig.update_traces(hovertemplate= None)
    fig.update_layout(xaxis_title="Hour", yaxis_title="Count",
                    #   xaxis={'categoryorder': 'category ascending',
                    #          'showgrid': False},
                    xaxis = {'showgrid': False, 'fixedrange': True},
                    yaxis={'showgrid': False, 'fixedrange': True},
                    plot_bgcolor='white',
                    legend_title=" ",
                    hovermode= 'x',
                    dragmode= False
                    )
    st.plotly_chart(fig, use_container_width=True, config=config)

    # new line
    st.markdown(f'<br>', unsafe_allow_html=True)

    #######################################
    # TBS Entrances For a Week
    #######################################
    df_weekly = df.copy()
    df_warnings.rename(columns= {'Type': 'Class'}, inplace= True)
    df_weekly = pd.concat([df_weekly, df_warnings])
    
    # change classes names
    df_weekly['Class'] = df_weekly['Class'].replace({'White_employee': 'Employee',
                                                     'Person': 'Customer',
                                                     'transaction':'Transaction',
                                                     'transaction_num':'Groups Cash',
                                                     'cluster': 'Groups Entry'})
    
    df_weekly['Week'] = df_weekly['Time'].dt.strftime('%Y-%U')
    df_weekly['Day'] = df_weekly['Time'].dt.strftime('%a')
    
    selected_week = day.strftime('%Y-%U')

    # Filter the DataFrame to get the whole week that matches the selected week
    filtered_df = df_weekly[df_weekly['Week'] == selected_week]
    
    grouped_dow = filtered_df.drop('Time', axis=1).groupby(
        ['Week', 'Day', 'Class']).sum().reset_index()
    days_order = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    grouped_dow['Day'] = pd.Categorical(
        grouped_dow['Day'], categories=days_order, ordered=True)
    grouped_dow_sorted = grouped_dow.sort_values('Day')
    pivot_grouped_dow = grouped_dow_sorted.pivot(
        index='Day', columns='Class', values='Count').reset_index()
    
    dropped_classes = ['Employee', 'missed opportunity', 'overstaffed', 'understaffed',
                        'long queue', 'hat missing food station', 'hat missing kitchen',
                        'pastry low', 'boxes low', 'salads low', 'spoilage',
                        'spoilageCount']
    
    for d_class in dropped_classes:
        try:
            pivot_grouped_dow = pivot_grouped_dow.drop(d_class, axis=1)
        except:
            continue
    # st.dataframe(pivot_grouped_dow)

    day_name = ''
    if pivot_grouped_dow.loc[pivot_grouped_dow["Customer"] == enter_stat.sum(), "Day"].values[0] == 'Sun':
        day_name = 'Sunday'
    elif pivot_grouped_dow.loc[pivot_grouped_dow["Customer"] == enter_stat.sum(), "Day"].values[0] == 'Mon':
        day_name = 'Monday'
    elif pivot_grouped_dow.loc[pivot_grouped_dow["Customer"] == enter_stat.sum(), "Day"].values[0] == 'Tue':
        day_name = 'Tuesday'
    elif pivot_grouped_dow.loc[pivot_grouped_dow["Customer"] == enter_stat.sum(), "Day"].values[0] == 'Wed':
        day_name = 'Wednesday'
    elif pivot_grouped_dow.loc[pivot_grouped_dow["Customer"] == enter_stat.sum(), "Day"].values[0] == 'Thu':
        day_name = 'Thursday'
    elif pivot_grouped_dow.loc[pivot_grouped_dow["Customer"] == enter_stat.sum(), "Day"].values[0] == 'Fri':
        day_name = 'Friday'
    elif pivot_grouped_dow.loc[pivot_grouped_dow["Customer"] == enter_stat.sum(), "Day"].values[0] == 'Sat':
        day_name = 'Saturday'

    st.markdown(
        f'<div class="container">' +
            '<p>TBS Entrances for a Week</p>' +
            '<div class="containerData">' +
            '<div class="boxData">' +
                '<p class="firstP">Week Total</p>' +
                f'<p class="secondP">{int(pivot_grouped_dow["Customer"].sum())}</p>' +
            '</div>' +
            '<div class="boxData">' +
                f'<p class="firstP">Total {day_name}</p>' +
                f'<p class="secondP">{int(enter_stat.sum())}</p>' +
            '</div>' +
            '</div>'
        '</div>'
        , unsafe_allow_html=True
    )

    # new line
    st.markdown(f'<br>', unsafe_allow_html=True)

    fig = px.line(pivot_grouped_dow, x="Day", y=pivot_grouped_dow.columns[1:],
                  color_discrete_map=colors
                  )
    fig.update_traces(hovertemplate= None)
    fig.update_layout(xaxis_title="Day of the Week", yaxis_title="Enteries", xaxis_tickformat='%a',
                      plot_bgcolor='white',
                      xaxis=dict(showgrid=False, fixedrange=True),
                      yaxis=dict(showgrid=False, fixedrange=True),
                      legend_title=" ",
                      hovermode= 'x',
                      dragmode= False
                      )
    st.plotly_chart(fig, use_container_width=True, config=config)
    # st.plotly_chart(fig, use_container_width=True)

    # new line
    # st.markdown(f'<br>', unsafe_allow_html=True)

    #######################################
    # TBS Complaince For a Week
    #######################################
    df_weekly_comp = df_warnings.copy()
    df_weekly_comp.rename(columns= {'Type': 'Class'}, inplace= True)
    df_weekly_comp = df_weekly_comp.loc[(df_weekly_comp['Class'] == 'missed opportunity') | 
                                        (df_weekly_comp['Class'] == 'overstaffed') |
                                        (df_weekly_comp['Class'] == 'understaffed') |
                                        (df_weekly_comp['Class'] == 'long queue') |
                                        (df_weekly_comp['Class'] == 'hat missing food station') |
                                        (df_weekly_comp['Class'] == 'hat missing kitchen') |
                                        (df_weekly_comp['Class'] == 'pastry low') |
                                        (df_weekly_comp['Class'] == 'boxes low') |
                                        (df_weekly_comp['Class'] == 'salads low')]
    
    # df_warnings.rename(columns= {'Type': 'Class'}, inplace= True)
    # df_weekly_comp = pd.concat([df_weekly_comp, df_warnings])
    
    # change classes names
    df_weekly_comp['Class'] = df_weekly_comp['Class'].replace({'missed opportunity': 'Unattended Client',
                                                     'overstaffed': 'Overstaffed',
                                                     'understaffed':'Understaffed',
                                                     'long queue':'Long Queue',
                                                     'hat missing food station':'Hat Missing Food Station',
                                                     'hat missing kitchen':'Hat Missing Kitchen',
                                                     'pastry low':'Pastry Low',
                                                     'boxes low':'Boxes Low',
                                                     'salads low':'Salads Low',
                                                     })
    
    df_weekly_comp['Week'] = df_weekly_comp['Time'].dt.strftime('%Y-%U')
    df_weekly_comp['Day'] = df_weekly_comp['Time'].dt.strftime('%a')
    
    selected_week = day.strftime('%Y-%U')

    # Filter the DataFrame to get the whole week that matches the selected week
    filtered_df = df_weekly_comp[df_weekly_comp['Week'] == selected_week]
    
    grouped_dow = filtered_df.drop('Time', axis=1).groupby(
        ['Week', 'Day', 'Class']).sum().reset_index()
    days_order = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    grouped_dow['Day'] = pd.Categorical(
        grouped_dow['Day'], categories=days_order, ordered=True)
    grouped_dow_sorted = grouped_dow.sort_values('Day')
    pivot_grouped_dow = grouped_dow_sorted.pivot(
        index='Day', columns='Class', values='Count').reset_index()

    st.markdown(
        f'<div class="container">' +
            '<p>TBS Compliance for a Week</p>' +
            '<div class="containerData">' +
            '<div class="boxData">' +
                '<p class="firstP">Week Total</p>' +
                f'<p class="secondP">{int(filtered_df["Count"].sum())}</p>' +
            '</div>' +
            '<div class="boxData">' +
                f'<p class="firstP">Total {day_name}</p>' +
                f'<p class="secondP">{int(filtered_df[filtered_df["Day"] == day_name[:3]]["Count"].sum())}</p>' +
            '</div>' +
            '</div>'
        '</div>'
        , unsafe_allow_html=True
    )

    # new line
    st.markdown(f'<br>', unsafe_allow_html=True)

    fig = px.line(pivot_grouped_dow, x="Day", y=pivot_grouped_dow.columns[1:],
                  color_discrete_map=colors
                  )
    fig.update_traces(hovertemplate= None)
    fig.update_layout(xaxis_title="Day of the Week", yaxis_title="Enteries", xaxis_tickformat='%a',
                      plot_bgcolor='white',
                      xaxis=dict(showgrid=False, fixedrange=True),
                      yaxis=dict(showgrid=False, fixedrange=True),
                      legend_title=" ",
                      hovermode= 'x',
                      dragmode= False
                      )
    st.plotly_chart(fig, use_container_width=True, config=config)
    # st.plotly_chart(fig, use_container_width=True)

    #######################################
    # TBS Hour Scheduler heatmap
    #######################################
    st.markdown(
        f'<p class="heatmapP" style="margin-bottom:0; position:relative; top:80px; z-index:6;">TBS Customers Heatmap</p>'
        , unsafe_allow_html=True
    )

    # new line
    # st.markdown(f'<br>', unsafe_allow_html=True)

    df_weekly = df.copy()
    df_weekly['Week'] = df_weekly['Time'].dt.strftime('%Y-%U')
    df_weekly['Day'] = df_weekly['Time'].dt.strftime('%a')
    df_weekly['Hour'] = df_weekly['Time'].dt.strftime('%I %p')
    # df_weekly['Hour'] = df_weekly['Time'].dt.hour

    selected_week = day.strftime('%Y-%U')

    # Filter the DataFrame to get the whole week that matches the selected week
    filtered_df = df_weekly[df_weekly['Week'] == selected_week]

    grouped_dow = filtered_df.drop('Time', axis=1).groupby(
        ['Week', 'Day', 'Hour', 'Class']).sum().reset_index()
    days_order = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    grouped_dow['Day'] = pd.Categorical(
        grouped_dow['Day'], categories=days_order, ordered=True)
    
    grouped_dow_sorted = grouped_dow.sort_values(by=['Day'])
    heatmap_data = grouped_dow_sorted[grouped_dow_sorted['Class'] == 'Person'].drop('Class', axis=1).pivot(index='Hour',
                                                                                                  columns='Day',
                                                                                                  values='Count').fillna(0)
    heatmap_data = heatmap_data.reset_index()
    # sorting dataframe
    heatmap_data[['x', 'y']] = heatmap_data['Hour'].str.split(n=1, expand=True)
    heatmap_data = heatmap_data.sort_values(by= ['y', 'x'], ignore_index = True)
    heatmap_data_sorted = pd.DataFrame(columns= heatmap_data.columns)
    
    try:
        last_am_index = heatmap_data[heatmap_data['y'] == 'AM'].index[-1]
        index_12pm = heatmap_data[heatmap_data['x'] == '12'].index[-1]
        heatmap_data_sorted = pd.concat([heatmap_data.iloc[:last_am_index + 1],
                                            heatmap_data.iloc[index_12pm: index_12pm + 1],
                                            heatmap_data.iloc[last_am_index + 1: -1]])
    except:
        heatmap_data_sorted = heatmap_data.copy()
    
    modified_heatmap_data = heatmap_data_sorted.set_index('Hour').drop(['x', 'y'], axis= 1)
    modified_heatmap_data = modified_heatmap_data[::-1]
    
    fig = go.Figure(data=go.Heatmap(
        z=modified_heatmap_data.values,
        x=modified_heatmap_data.columns.values,
        y=modified_heatmap_data.index.values,
        colorscale='YlGn',
        hovertemplate='%{x}<br>%{y}<br>Count: %{z}<extra></extra>'))

    for i in range(len(modified_heatmap_data.index)):
        for j in range(len(modified_heatmap_data.columns)):
            cell_value = modified_heatmap_data.values[i, j]
            if cell_value > modified_heatmap_data.values.max() / 2:
                text_color = 'white'
            else:
                text_color = 'black'

            fig.add_annotation(
                x=modified_heatmap_data.columns[j], y=modified_heatmap_data.index[i],
                text=str(int(cell_value)),
                showarrow=False,
                font=dict(color=text_color)
            )
    fig.update_layout(
        xaxis_title="Day of the Week",
        yaxis_title="Hour",
        xaxis_nticks=len(modified_heatmap_data.columns),
        yaxis_nticks=len(modified_heatmap_data.index) + 1,
        xaxis_tickformat='%b %d',
        xaxis_tickvals=modified_heatmap_data.columns.values,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        dragmode= False
        
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

    #######################################
    # TBS Hour Scheduler heatmap
    #######################################
    st.markdown(
        f'<p class="heatmapP" style="margin-bottom:0; position:relative; top:80px; z-index:6;">TBS Employees Heatmap</p>'
        , unsafe_allow_html=True
    )

    df_weekly = df.copy()
    df_weekly['Week'] = df_weekly['Time'].dt.strftime('%Y-%U')
    df_weekly['Day'] = df_weekly['Time'].dt.strftime('%a')
    df_weekly['Hour'] = df_weekly['Time'].dt.strftime('%I %p')


    selected_week = day.strftime('%Y-%U')

    # Filter the DataFrame to get the whole week that matches the selected week
    filtered_df = df_weekly[df_weekly['Week'] == selected_week]

    grouped_dow = filtered_df.drop('Time', axis=1).groupby(
        ['Week', 'Day', 'Hour', 'Class']).sum().reset_index()
    days_order = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    grouped_dow['Day'] = pd.Categorical(
        grouped_dow['Day'], categories=days_order, ordered=True)
    
    grouped_dow_sorted = grouped_dow.sort_values(by=['Day'])
    heatmap_data = grouped_dow_sorted[grouped_dow_sorted['Class'] == 'White_employee'].drop('Class', axis=1).pivot(index='Hour',
                                                                                                  columns='Day',
                                                                                                  values='Count').fillna(0)
    heatmap_data = heatmap_data.reset_index()

    # sorting dataframe
    heatmap_data[['x', 'y']] = heatmap_data['Hour'].str.split(n=1, expand=True)
    heatmap_data = heatmap_data.sort_values(by= ['y', 'x'], ignore_index = True)
    heatmap_data_sorted = pd.DataFrame(columns= heatmap_data.columns)
    
    try:
        last_am_index = heatmap_data[heatmap_data['y'] == 'AM'].index[-1]
        index_12pm = heatmap_data[heatmap_data['x'] == '12'].index[-1]
        heatmap_data_sorted = pd.concat([heatmap_data.iloc[:last_am_index + 1],
                                            heatmap_data.iloc[index_12pm: index_12pm + 1],
                                            heatmap_data.iloc[last_am_index + 1: -1]])
    except:
        heatmap_data_sorted = heatmap_data.copy()
    
    # st.dataframe(heatmap_data)
    # st.dataframe(heatmap_data_sorted)
    modified_heatmap_data = heatmap_data_sorted.set_index('Hour').drop(['x', 'y'], axis= 1)
    modified_heatmap_data = modified_heatmap_data[::-1]
    # st.dataframe(modified_heatmap_data[::-1])
    fig = go.Figure(data=go.Heatmap(
        z=modified_heatmap_data.values,
        x=modified_heatmap_data.columns.values,
        y=modified_heatmap_data.index.values,
        colorscale='YlGn',
        hovertemplate='%{x}<br>%{y}<br>Count: %{z}<extra></extra>'))

    for i in range(len(modified_heatmap_data.index)):
        for j in range(len(modified_heatmap_data.columns)):
            cell_value = modified_heatmap_data.values[i, j]
            if cell_value > modified_heatmap_data.values.max() / 2:
                text_color = 'white'
            else:
                text_color = 'black'

            fig.add_annotation(
                x=modified_heatmap_data.columns[j], y=modified_heatmap_data.index[i],
                text=str(int(cell_value)),
                showarrow=False,
                font=dict(color=text_color)
            )
    fig.update_layout(
        xaxis_title="Day of the Week",
        yaxis_title="Hour",
        xaxis_nticks=len(modified_heatmap_data.columns),
        yaxis_nticks=len(modified_heatmap_data.index) + 1,
        xaxis_tickformat='%b %d',
        xaxis_tickvals=modified_heatmap_data.columns.values,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        dragmode= False
    )

    st.plotly_chart(fig, use_container_width=True, config=config)
    # st.plotly_chart(fig, use_container_width=True)

elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
