# Colombo Apartment Price Predictor
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st

st.set_page_config(page_title='Colombo Apartment Price Predictor', page_icon='🏢', layout='wide')

@st.cache_resource
def load():
    return (joblib.load('models.joblib'), joblib.load('meta.joblib'),
            pd.read_csv('model_results.csv'), pd.read_csv('shap_importance.csv'))

models, meta, results, shap_imp = load()

@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(models['LightGBM'])

KW_LABELS = {'is_luxury': 'Luxury', 'is_furnished': 'Furnished', 'is_brand_new': 'Brand new',
             'is_penthouse': 'Penthouse', 'is_sea_view': 'Sea view',
             'is_landmark': 'Landmark project (Cinnamon Life, Altair, Tri-Zen, Havelock City…)'}

def build_row(zone, area, beds, baths, flags):
    row = pd.DataFrame(0, index=[0], columns=meta['columns'], dtype=float)
    row['area_sqft'], row['bedrooms'], row['bathrooms'] = area, beds, baths
    row['bath_per_bed'] = baths / beds
    for k, v in flags.items():
        row[k] = int(v)
    row[meta['zone_cols'][zone]] = 1
    return row

def nice_name(col):
    if col.startswith('zone_'):
        return 'Zone: ' + col[5:].replace('_', ' ').strip()
    return KW_LABELS.get(col, col).split(' (')[0]

# ---------------- Sidebar inputs ----------------
st.sidebar.header('🏢 Apartment details')
zones = list(meta['zone_cols'])
zone = st.sidebar.selectbox('Location zone', zones,
                            index=zones.index('Colombo 05') if 'Colombo 05' in zones else 0)
area = st.sidebar.number_input('Floor area (sq ft)', 300, 10000, 1200, step=50)
beds = st.sidebar.slider('Bedrooms', 1, 6, 3)
baths = st.sidebar.slider('Bathrooms', 1, 6, 2)
st.sidebar.markdown('**Ad features**')
flags = {k: st.sidebar.checkbox(KW_LABELS[k]) for k in meta['keywords']}

row = build_row(zone, area, beds, baths, flags)

st.title('Colombo Apartment Price Predictor')
st.caption(f"Ensemble learning (bagging · boosting · stacking) on {meta['n_listings']:,} "
           "ikman.lk apartment listings")

tab1, tab2, tab3, tab4 = st.tabs(['💰 Predict', '📊 Model comparison', '🔍 What drives price', 'ℹ️ About'])

# ---------------- Tab 1: Predict ----------------
with tab1:
    preds = {name: float(np.exp(m.predict(row)[0])) for name, m in models.items()}
    main = preds['Stacking (best)']
    err = meta['mape'] / 100

    c1, c2, c3 = st.columns(3)
    c1.metric('Predicted price (Stacking)', f'Rs {main / 1e6:,.1f} M')
    c2.metric('Likely range (± typical error)',
              f'Rs {main * (1 - err) / 1e6:,.0f} – {main * (1 + err) / 1e6:,.0f} M')
    c3.metric(f'Median listing in {zone}', f"Rs {meta['zone_median_price'][zone] / 1e6:,.0f} M")
    st.caption(f'Price per sq ft: Rs {main / area:,.0f}')

    st.subheader('What each model predicts')
    comp = (pd.DataFrame({'Model': list(preds), 'Predicted (Rs M)': [v / 1e6 for v in preds.values()]})
            .set_index('Model').round(1))
    st.bar_chart(comp)
    st.dataframe(comp, use_container_width=True)
    st.info('Models disagree most on unusual flats (very large, landmark or penthouse units). '
            'That disagreement is a useful signal of uncertainty.')

# ---------------- Tab 2: Model comparison ----------------
with tab2:
    st.subheader('Test-set results (405 unseen listings)')
    st.dataframe(results, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**R² on log price (higher = better)**')
        st.bar_chart(results.set_index('Model')['R² (log)'])
    with c2:
        st.markdown('**Mean % error (lower = better)**')
        st.bar_chart(results.set_index('Model')['MAPE %'])
    st.markdown("""
**Key findings**
- **Boosting beat bagging**: sequential trees reduce *bias*; bagging mainly reduces *variance*.
- **Stacking was best, but only slightly**: the base models make similar (correlated) errors.
- **RF + ANN hybrid had the lowest rupee MAE**: the two learners make *different* kinds of errors.
- **AdaBoost was worst**: it up-weights mispriced ads and chases noise.
""")

# ---------------- Tab 3: SHAP ----------------
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Overall: what drives price')
        top = shap_imp.head(12).copy()
        top['feature'] = top['feature'].map(nice_name)
        st.bar_chart(top.set_index('feature')['mean_abs_shap'])
        st.caption('Mean |SHAP value| from LightGBM across the test set.')
    with c2:
        st.subheader('This apartment: why this price?')
        sv = get_explainer().shap_values(row)[0]
        contrib = pd.DataFrame({'feature': row.columns, 'shap': sv, 'value': row.iloc[0].values})
        contrib = contrib[(contrib['value'] != 0) | ~contrib['feature'].str.startswith('zone_')]
        contrib = contrib.reindex(contrib['shap'].abs().sort_values(ascending=False).index).head(8)
        contrib['Effect on price (%)'] = ((np.exp(contrib['shap']) - 1) * 100).round(1)
        contrib['feature'] = contrib['feature'].map(nice_name)
        st.bar_chart(contrib.set_index('feature')['Effect on price (%)'])
        st.caption('Positive = pushes price up vs an average listing (LightGBM explanation).')

# ---------------- Tab 4: About ----------------
with tab4:
    st.markdown(f"""
**Data:** {meta['n_listings']:,} apartment-for-sale listings in the Colombo area, scraped from ikman.lk
(Aug–Sep). Cleaned to remove rentals, invalid values and duplicate re-posts. Location zones were
extracted from the ad text.

**Models:** Linear Regression (baseline) · Bagging · Random Forest · AdaBoost · Gradient Boosting ·
XGBoost · LightGBM · CatBoost · Stacking (RF + XGB + LGBM + CatBoost → Ridge) · ANN · Hybrid RF + ANN.

**Limitations:** asking prices, not sale prices; Colombo area only; no floor level, age or amenities.

**Used the help of AI for the development of the dashboard**

""")
