import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DARK   = '#0d1117'
ACCENT = '#58a6ff'
RED    = '#f85149'
GREEN  = '#3fb950'
YELLOW = '#d29922'
PURPLE = '#bc8cff'
WHITE  = '#e6edf3'
GREY   = '#21262d'

def generate_covid_data():
    np.random.seed(42)
    countries = {
        'USA':          {'pop': 331000000,  'base_cases': 100000000, 'base_deaths': 1100000, 'vax_rate': 0.68},
        'India':        {'pop': 1380000000, 'base_cases': 44000000,  'base_deaths': 530000,  'vax_rate': 0.67},
        'Brazil':       {'pop': 214000000,  'base_cases': 37000000,  'base_deaths': 700000,  'vax_rate': 0.79},
        'France':       {'pop': 67000000,   'base_cases': 39000000,  'base_deaths': 168000,  'vax_rate': 0.78},
        'Germany':      {'pop': 83000000,   'base_cases': 38000000,  'base_deaths': 174000,  'vax_rate': 0.76},
        'UK':           {'pop': 67000000,   'base_cases': 24000000,  'base_deaths': 220000,  'vax_rate': 0.70},
        'Russia':       {'pop': 144000000,  'base_cases': 22000000,  'base_deaths': 393000,  'vax_rate': 0.53},
        'Italy':        {'pop': 60000000,   'base_cases': 26000000,  'base_deaths': 190000,  'vax_rate': 0.80},
        'Spain':        {'pop': 47000000,   'base_cases': 13000000,  'base_deaths': 120000,  'vax_rate': 0.82},
        'Argentina':    {'pop': 45000000,   'base_cases': 10000000,  'base_deaths': 130000,  'vax_rate': 0.87},
        'Mexico':       {'pop': 126000000,  'base_cases': 7400000,   'base_deaths': 334000,  'vax_rate': 0.61},
        'South Africa': {'pop': 60000000,   'base_cases': 4000000,   'base_deaths': 102000,  'vax_rate': 0.32},
        'Japan':        {'pop': 125000000,  'base_cases': 32000000,  'base_deaths': 73000,   'vax_rate': 0.84},
        'Australia':    {'pop': 26000000,   'base_cases': 11000000,  'base_deaths': 22000,   'vax_rate': 0.86},
        'Canada':       {'pop': 38000000,   'base_cases': 4500000,   'base_deaths': 51000,   'vax_rate': 0.83},
    }
    dates = pd.date_range('2020-01-01', '2023-06-30', freq='W')
    rows = []
    for country, stats in countries.items():
        n = len(dates)
        wave1 = np.exp(-((np.arange(n) - 15)**2)  / 200) * 0.15
        wave2 = np.exp(-((np.arange(n) - 45)**2)  / 300) * 0.25
        wave3 = np.exp(-((np.arange(n) - 80)**2)  / 400) * 0.35
        wave4 = np.exp(-((np.arange(n) - 110)**2) / 500) * 0.25
        combined = wave1 + wave2 + wave3 + wave4
        combined = combined / combined.sum()
        total_cases  = (np.cumsum(combined) * stats['base_cases']).astype(int)
        total_deaths = (np.cumsum(combined) * stats['base_deaths']).astype(int)
        vax_ramp  = np.clip((np.arange(n) - 55) / 80, 0, 1)
        total_vax = (vax_ramp * stats['vax_rate'] * stats['pop']).astype(int)
        new_cases  = np.diff(total_cases,  prepend=0)
        new_deaths = np.diff(total_deaths, prepend=0)
        for i, d in enumerate(dates):
            rows.append({
                'location':           country,
                'date':               d,
                'total_cases':        total_cases[i],
                'new_cases':          max(0, new_cases[i]),
                'total_deaths':       total_deaths[i],
                'new_deaths':         max(0, new_deaths[i]),
                'total_vaccinations': total_vax[i],
                'population':         stats['pop'],
            })
    df = pd.DataFrame(rows)
    df['cases_per_million']  = (df['total_cases']  / df['population'] * 1e6).round(1)
    df['deaths_per_million'] = (df['total_deaths'] / df['population'] * 1e6).round(1)
    df['vax_per_hundred']    = (df['total_vaccinations'] / df['population'] * 100).round(2)
    df['cfr']                = (df['total_deaths'] / df['total_cases'].replace(0, 1) * 100).round(3)
    return df

print("=" * 60)
print("DATA VISUALIZATION: COVID-19 GLOBAL DATA")
print("=" * 60)

df = generate_covid_data()
df.to_csv("covid_data.csv", index=False)
print(f"Dataset ready: {len(df)} rows across {df['location'].nunique()} countries")
print(f"Date range : {df['date'].min().date()} -> {df['date'].max().date()}")

latest  = df.sort_values('date').groupby('location').last().reset_index()
monthly = df.copy()
monthly['month'] = monthly['date'].dt.to_period('M')
monthly_global   = monthly.groupby('month')['new_cases'].sum().reset_index()
monthly_global['month_dt'] = monthly_global['month'].dt.to_timestamp()

def style_ax(ax, title):
    ax.set_facecolor(GREY)
    ax.set_title(title, color=WHITE, fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

sns.set_theme(style='dark')
fig = plt.figure(figsize=(20, 22), facecolor=DARK)
fig.suptitle('COVID-19 Global Data Analysis (2020–2023)',
             fontsize=22, fontweight='bold', color=WHITE, y=0.99)

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                      left=0.07, right=0.97, top=0.95, bottom=0.05)

ax1 = fig.add_subplot(gs[0, 0])
top10_cases = latest.nlargest(10, 'total_cases')
colors_bar  = [ACCENT if i == 0 else '#1f4e8c' for i in range(len(top10_cases))]
bars = ax1.barh(top10_cases['location'][::-1],
                top10_cases['total_cases'][::-1] / 1e6,
                color=colors_bar[::-1], edgecolor=DARK, linewidth=0.5)
style_ax(ax1, 'Top 10 Countries — Total Cases (Millions)')
ax1.set_xlabel('Total Cases (Millions)', color=WHITE)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}M'))
for bar in bars:
    w = bar.get_width()
    ax1.text(w + 0.3, bar.get_y() + bar.get_height()/2,
             f'{w:.1f}M', va='center', color=WHITE, fontsize=8)

ax2 = fig.add_subplot(gs[0, 1])
top10_deaths = latest.nlargest(10, 'total_deaths')
colors_d     = [RED if i == 0 else '#7f1d1d' for i in range(len(top10_deaths))]
ax2.barh(top10_deaths['location'][::-1],
         top10_deaths['total_deaths'][::-1] / 1e3,
         color=colors_d[::-1], edgecolor=DARK, linewidth=0.5)
style_ax(ax2, 'Top 10 Countries — Total Deaths (Thousands)')
ax2.set_xlabel('Total Deaths (Thousands)', color=WHITE)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}K'))

ax3 = fig.add_subplot(gs[0, 2])
scatter = ax3.scatter(latest['cases_per_million'],
                      latest['deaths_per_million'],
                      c=latest['vax_per_hundred'],
                      cmap='RdYlGn', s=120,
                      edgecolors=WHITE, linewidths=0.5, alpha=0.9)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Vaccination %', color=WHITE, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)
for _, row in latest.iterrows():
    ax3.annotate(row['location'],
                 (row['cases_per_million'], row['deaths_per_million']),
                 fontsize=7, color=WHITE, alpha=0.85,
                 xytext=(4, 4), textcoords='offset points')
style_ax(ax3, 'Cases vs Deaths per Million\n(Color = Vaccination Rate)')
ax3.set_xlabel('Cases per Million', color=WHITE)
ax3.set_ylabel('Deaths per Million', color=WHITE)

ax4 = fig.add_subplot(gs[1, :2])
ax4.fill_between(monthly_global['month_dt'],
                 monthly_global['new_cases'] / 1e6,
                 color=ACCENT, alpha=0.3)
ax4.plot(monthly_global['month_dt'],
         monthly_global['new_cases'] / 1e6,
         color=ACCENT, linewidth=2)
waves = [('2020-04', 'Wave 1'), ('2020-12', 'Wave 2'),
         ('2021-09', 'Delta'),  ('2022-01', 'Omicron')]
for wdate, wlabel in waves:
    xpos = pd.Timestamp(wdate)
    yval = monthly_global[monthly_global['month_dt'] >= xpos]['new_cases'].iloc[0] / 1e6
    ax4.axvline(xpos, color=YELLOW, linestyle='--', linewidth=1, alpha=0.7)
    ax4.text(xpos, yval + 0.5, wlabel, color=YELLOW, fontsize=8, rotation=30)
style_ax(ax4, 'Global Weekly New Cases Over Time (Millions)')
ax4.set_xlabel('Date', color=WHITE)
ax4.set_ylabel('New Cases (Millions)', color=WHITE)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}M'))

ax5 = fig.add_subplot(gs[1, 2])
top_vax = latest.nlargest(10, 'vax_per_hundred')[['location', 'vax_per_hundred']]
bar_colors_v = [GREEN if v >= 80 else YELLOW if v >= 60 else RED
                for v in top_vax['vax_per_hundred']]
ax5.barh(top_vax['location'][::-1], top_vax['vax_per_hundred'][::-1],
         color=bar_colors_v[::-1], edgecolor=DARK)
ax5.axvline(70, color=WHITE, linestyle='--', linewidth=1,
            alpha=0.5, label='70% target')
style_ax(ax5, 'Vaccination Rate by Country (%)')
ax5.set_xlabel('Vaccinations per 100 People', color=WHITE)
ax5.legend(facecolor=GREY, labelcolor=WHITE, fontsize=8)
for i, (_, row) in enumerate(top_vax[::-1].iterrows()):
    ax5.text(row['vax_per_hundred'] + 0.5, i,
             f"{row['vax_per_hundred']:.0f}%", va='center', color=WHITE, fontsize=8)

ax6 = fig.add_subplot(gs[2, 0])
cfr_data = latest[['location', 'cfr']].sort_values('cfr', ascending=False).head(10)
ax6.bar(cfr_data['location'], cfr_data['cfr'],
        color=[RED if c > 3 else YELLOW if c > 1.5 else GREEN for c in cfr_data['cfr']],
        edgecolor=DARK)
style_ax(ax6, 'Case Fatality Rate by Country (%)')
ax6.set_ylabel('CFR (%)', color=WHITE)
ax6.tick_params(axis='x', rotation=35)
ax6.axhline(cfr_data['cfr'].mean(), color=WHITE, linestyle='--',
            linewidth=1, alpha=0.6, label=f"Avg: {cfr_data['cfr'].mean():.1f}%")
ax6.legend(facecolor=GREY, labelcolor=WHITE, fontsize=8)

ax7 = fig.add_subplot(gs[2, 1])
key_countries = ['USA', 'India', 'Brazil', 'UK', 'Japan']
palette       = [ACCENT, GREEN, RED, YELLOW, PURPLE]
for country, color in zip(key_countries, palette):
    cdf = df[df['location'] == country].copy()
    cdf['rolling_new'] = cdf['new_cases'].rolling(4).mean()
    ax7.plot(cdf['date'], cdf['rolling_new'] / 1e3,
             label=country, color=color, linewidth=1.8, alpha=0.9)
style_ax(ax7, 'New Cases Over Time — Key Countries\n(4-week rolling avg, Thousands)')
ax7.set_xlabel('Date', color=WHITE)
ax7.set_ylabel('New Cases (Thousands)', color=WHITE)
ax7.legend(facecolor=GREY, labelcolor=WHITE, fontsize=8)
ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}K'))

ax8 = fig.add_subplot(gs[2, 2])
regions = {
    'Americas': ['USA', 'Brazil', 'Argentina', 'Mexico', 'Canada'],
    'Europe':   ['France', 'Germany', 'UK', 'Italy', 'Spain', 'Russia'],
    'Asia':     ['India', 'Japan'],
    'Africa':   ['South Africa'],
    'Oceania':  ['Australia'],
}
region_totals = {region: latest[latest['location'].isin(countries)]['total_cases'].sum()
                 for region, countries in regions.items()}
region_df = pd.Series(region_totals).sort_values(ascending=False)
wedges, texts, autotexts = ax8.pie(
    region_df.values,
    labels=region_df.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=[ACCENT, RED, GREEN, YELLOW, PURPLE],
    wedgeprops=dict(edgecolor=DARK, linewidth=1.5),
    textprops=dict(color=WHITE, fontsize=9)
)
for at in autotexts:
    at.set_color(WHITE)
    at.set_fontweight('bold')
ax8.set_facecolor(GREY)
ax8.set_title('Total Cases by Region', color=WHITE, fontsize=13, fontweight='bold', pad=10)

plt.savefig('covid_visualization.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.show()
print("\nPlot saved as 'covid_visualization.png'")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"  Countries tracked         : {df['location'].nunique()}")
print(f"  Date range                : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Highest cases             : {latest.nlargest(1,'total_cases')['location'].values[0]}")
print(f"  Highest deaths            : {latest.nlargest(1,'total_deaths')['location'].values[0]}")
print(f"  Highest vaccination rate  : {latest.nlargest(1,'vax_per_hundred')['location'].values[0]} "
      f"({latest['vax_per_hundred'].max():.0f}%)")
print(f"  Lowest  vaccination rate  : {latest.nsmallest(1,'vax_per_hundred')['location'].values[0]} "
      f"({latest['vax_per_hundred'].min():.0f}%)")
print(f"  Highest CFR               : {latest.nlargest(1,'cfr')['location'].values[0]} "
      f"({latest['cfr'].max():.2f}%)")
print("=" * 60)
