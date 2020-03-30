# import bootstrap

import helper

help = helper.helper()
help.properties()
config = help.config

print(config)





base_list, params, p_value, min_param, category = [], [], [], [], []
import preprocess
def iterator(df, x, y=['disc, time, dow']) :
    for i in range(len(np.unique(df.x))) :
        # x = prod_cd
        iter_data = df[df.x == np.unique(df.x)[i]].iloc[:, 1:]
        temp_code = np.unique(df.x)[i]
        for s in range(len(y)) :
            min_param = np.argmin(df.groupby([y[s]])['normal_y'].mean())
            category.append(y[s])



    disc_name_list.append(np.unique(df_a.tot_disc))

    result = ols("sell_qty ~ C(time, Treatment(" + str(min_idx_time) + ")) + C(dow, Treatment(" + str(
        min_idx_dow) + "))+ C(tot_disc, Treatment(" + str(min_idx_disc) + "))", data=df_a).fit()
    params.append([result.params])
    p_val.append([result.pvalues])
    min_time.append(min_idx_time)
    min_dow.append(min_idx_dow)
    min_disc.append(min_idx_disc)
    # sku 기준 생성
    sku_l.append(prod_cd)

# %%
##### 5. regression result to dataframe
# 계수
df_time, df_dow, df_disc = [], [], []
# p_value
pval_time, pval_dow, pval_disc = [], [], []
# 시간
time_val, dow_val, disc_val = [], [], []

for seq in range(len(sku_l)):
    # coeffients 값 구하기
    df_time.append(param_time(min_idx_time, params, seq)[0])
    df_dow.append(param_dow(min_idx_dow, params, seq)[0])
    df_disc.append(param_disc(min_idx_disc, params, seq)[0])
    # p_value 값 구하기
    pval_time.append(param_time(min_idx_time, p_val, seq)[0])
    pval_dow.append(param_dow(min_idx_dow, p_val, seq)[0])
    pval_disc.append(param_disc(min_idx_disc, p_val, seq)[0])
    # 시간 구하기
    time_val.append(param_time(min_idx_time, params, seq)[1])
    dow_val.append(param_dow(min_idx_dow, params, seq)[1])
    disc_val.append(param_disc(min_idx_disc, params, seq)[1])

    # %%
