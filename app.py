import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components

# 1. 웹 브라우저 탭 설정 및 와이드 레이아웃 적용
st.set_page_config(
    page_title="대한민국 교통사고 통합 분석 플랫폼",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 대한민국 교통사고 데이터 분석 및 예측 대시보드")
st.write("공공데이터 기반의 공간 히트맵 시각화와 머신러닝 예측 시뮬레이션을 실시간으로 수행하는 1인 전문가 플랫폼입니다.")

# 2. 데이터 캐싱 로드 함수 (컬럼명 공백 제거 전처리 포함)
@st.cache_data
def load_traffic_data():
    file_name = "시도 시군구별 교통사고 통계.csv"
    df = pd.read_csv(file_name, encoding='euc-kr')
    
    # 🌟 핵심 방어 코드: 컬럼명 양옆의 보이지 않는 공백 제거
    df.columns = df.columns.str.strip()
    return df

df = load_traffic_data()

# 3. 사이드바 메뉴바 구성
st.sidebar.header("🔍 분석 메뉴 선택")
menu = st.sidebar.selectbox("이동할 페이지를 선택하세요", ["데이터 & 상관관계 분석", "교통사고 히트맵 지도", "AI 부상자수 예측 시뮬레이터"])

# 4. 메뉴별 화면 렌더링
if menu == "데이터 & 상관관계 분석":
    st.subheader("📊 원본 공공데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📈 수치형 변수 간 상관관계 매트릭스")
    corr_matrix = df.select_dtypes(include=['number']).corr()
    st.dataframe(corr_matrix.style.format("{:.3f}"), use_container_width=True)

elif menu == "교통사고 히트맵 지도":
    st.subheader("🔥 대한민국 시도별 교통사고 밀도 히트맵")
    st.write("사고 밀도가 높은 지역은 붉은색, 낮은 지역은 푸른색 계열의 열 지도(Heatmap)로 시각화됩니다.")
    
    base_map = folium.Map(location=[36.5, 127.5], zoom_start=7)
    
    coordinates = {
        '서울': [37.5665, 126.9780], '부산': [35.1796, 129.0756], '대구': [35.8714, 128.6014],
        '인천': [37.4563, 126.7052], '광주': [35.1595, 126.8526], '대전': [36.3504, 127.3845],
        '울산': [35.5384, 129.3114], '경기': [37.4138, 127.5183], '강원': [37.8228, 128.1555],
        '충북': [36.6356, 127.4912], '충남': [36.5184, 126.8000], '전북': [35.7175, 127.1530],
        '전남': [34.8161, 126.4629], '경북': [36.5760, 128.5057], '경남': [35.4606, 128.2132],
        '제주': [33.4996, 126.5312], '세종': [36.4800, 127.2890]
    }
    
    heat_data = []
    for idx, row in df.iterrows():
        sido = row['시도']
        cnt = row['사고건수']
        if sido in coordinates:
            heat_data.append([coordinates[sido][0], coordinates[sido][1], cnt])
            
    HeatMap(heat_data, radius=15, blur=10).add_to(base_map)
    
    components.html(base_map._repr_html_(), height=550)

elif menu == "AI 부상자수 예측 시뮬레이터":
    st.subheader("🤖 선형 회귀(Linear Regression) 기반 사고 예측 엔진")
    st.write("학습된 머신러닝 모델을 활용하여, 임의의 사고건수 발생 시 예상되는 부상자수를 실시간으로 시뮬레이션합니다.")
    
    # 🌟 동적 컬럼 선택 기능 추가: KeyError 원천 봉쇄
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("독립 변수 (원인 데이터)", options=numeric_cols, index=0 if '사고건수' not in numeric_cols else numeric_cols.index('사고건수'))
    with col2:
        y_col = st.selectbox("종속 변수 (결과 예측 데이터)", options=numeric_cols, index=0 if '부상자수' not in numeric_cols else numeric_cols.index('부상자수'))
        
    X = df[[x_col]]
    y = df[y_col]
    
    model = LinearRegression()
    model.fit(X, y)
    
    user_input = st.number_input(f"예상할 {x_col}를 입력하세요:", min_value=1, max_value=1000000, value=5000, step=100)
    
    if st.button("🔮 미래 예측 실행"):
        predicted = model.predict([[user_input]])
        pred_value = float(predicted.flatten()[0])
        st.success(f"✨ 입력하신 **{x_col} {user_input:,}** 기준, 예상되는 **{y_col}**는 약 **{pred_value:,.1f}**입니다.")