#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票量价策略回测系统
基于akshare和vectorbt实现沪深500策略回测
"""

import akshare as ak
import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置vectorbt全局配置
vbt.settings.array_wrapper['freq'] = 'D'  # 设置默认频率为日频

class VolumePrice_Strategy:
    """量价策略类"""

    def __init__(self, initial_capital=1000000, position_size=0.05,
                 max_holding_days=3, trail_percent=0.03,
                 stop_loss_pct=0.02, profit_take_pct=0.05):
        """初始化策略

        :param initial_capital: 初始资金，默认100万
        :param position_size: 单只股票仓位
        :param max_holding_days: 最大持仓天数
        :param trail_percent: 移动止损百分比
        :param stop_loss_pct: 固定止损百分比
        :param profit_take_pct: 止盈百分比
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_holding_days = max_holding_days
        self.trail_percent = trail_percent
        self.stop_loss_pct = stop_loss_pct
        self.profit_take_pct = profit_take_pct
        
    def get_hs500_stocks(self):
        """获取沪深500成分股"""
        try:
            print("正在获取沪深500成分股...")
            # 获取沪深500成分股
            hs500_stocks = ak.index_stock_cons(symbol="000905")  # 中证500指数代码
            print(f"成功获取{len(hs500_stocks)}只中证500成分股")
            return hs500_stocks['品种代码'].tolist()
        except Exception as e:
            print(f"获取中证500成分股失败: {e}")
            try:
                # 尝试获取沪深300成分股
                hs300_stocks = ak.index_stock_cons(symbol="000300")
                print(f"改为获取{len(hs300_stocks)}只沪深300成分股")
                return hs300_stocks['品种代码'].tolist()
            except Exception as e2:
                print(f"获取沪深300成分股也失败: {e2}")
                # 如果都获取失败，使用一些示例股票代码
                return ['000001', '000002', '000858', '002415', '600000', '600036', '600519', '600887']
    
    def get_stock_data(self, stock_code, start_date, end_date):
        """
        获取股票数据
        :param stock_code: 股票代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 股票数据DataFrame
        """
        try:
            # 获取股票历史数据
            stock_data = ak.stock_zh_a_hist(symbol=stock_code, 
                                          start_date=start_date.replace('-', ''), 
                                          end_date=end_date.replace('-', ''),
                                          adjust="qfq")  # 前复权
            
            if stock_data.empty:
                return None
            
            print(f"股票{stock_code}原始数据列数: {len(stock_data.columns)}")
            print(f"股票{stock_code}原始列名: {list(stock_data.columns)}")
            
            # 根据实际列数动态处理
            if len(stock_data.columns) == 12:
                # 12列的情况：['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                stock_data.columns = ['date', 'stock_code', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'change_pct', 'change_amount', 'turnover_rate']
                # 删除股票代码列，因为我们不需要它
                stock_data = stock_data.drop('stock_code', axis=1)
            elif len(stock_data.columns) == 11:
                # 11列的情况
                stock_data.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'change_pct', 'change_amount', 'turnover_rate']
            elif len(stock_data.columns) == 10:
                # 10列的情况，可能缺少某一列
                stock_data.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'change_pct', 'change_amount']
                stock_data['turnover_rate'] = 0  # 添加缺失列
            else:
                # 其他情况，使用默认列名
                print(f"警告：股票{stock_code}列数为{len(stock_data.columns)}，使用默认处理")
                # 确保至少有基本的OHLCV数据
                required_cols = ['date', 'open', 'close', 'high', 'low', 'volume']
                if len(stock_data.columns) >= 6:
                    stock_data.columns = required_cols + [f'col_{i}' for i in range(6, len(stock_data.columns))]
                else:
                    print(f"股票{stock_code}数据列数不足，跳过")
                    return None
            
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data.set_index('date', inplace=True)
            
            # 不强制设置频率，让pandas自动推断
            # stock_data.index.freq = 'B'  # Business day frequency
            
            # 计算移动平均线
            stock_data['ma5'] = stock_data['close'].rolling(window=5).mean()
            stock_data['ma10'] = stock_data['close'].rolling(window=10).mean()
            stock_data['ma20'] = stock_data['close'].rolling(window=20).mean()
            stock_data['ma60'] = stock_data['close'].rolling(window=60).mean()
            
            # 计算成交量相关指标
            stock_data['volume_ma22'] = stock_data['volume'].rolling(window=22).mean()  # 近一个月平均成交量
            stock_data['volume_ma252'] = stock_data['volume'].rolling(window=252).mean()  # 近一年平均成交量
            stock_data['volume_std252'] = stock_data['volume'].rolling(window=252).std()  # 近一年成交量标准差
            
            # 如果没有change_pct列，计算涨跌幅
            if 'change_pct' not in stock_data.columns:
                stock_data['change_pct'] = stock_data['close'].pct_change() * 100
            
            return stock_data
            
        except Exception as e:
            print(f"获取股票{stock_code}数据失败: {e}")
            return None
    
    def check_volume_condition(self, data):
        """
        检查量能条件
        :param data: 股票数据
        :return: 是否满足量能条件
        """
        if len(data) < 100:  # 降低数据要求
            return False
            
        latest_data = data.iloc[-22:]  # 最近一个月
        
        # 降低爆量条件：最近一月成交量平均值 > (过去一年成交量平均值 + 1×过去一年成交量标准差)
        recent_volume_avg = latest_data['volume'].mean()
        if len(data) >= 252:
            yearly_volume_avg = data['volume_ma252'].iloc[-1]
            yearly_volume_std = data['volume_std252'].iloc[-1]
            volume_threshold = yearly_volume_avg + 1 * yearly_volume_std  # 降低阈值
        else:
            # 如果数据不足一年，使用全部数据计算
            yearly_volume_avg = data['volume'].mean()
            yearly_volume_std = data['volume'].std()
            volume_threshold = yearly_volume_avg + 1 * yearly_volume_std
        
        is_volume_surge = recent_volume_avg > volume_threshold
        
        # 降低涨停条件：近期是否出现过大涨（涨幅>=5%）
        has_big_rise = (latest_data['change_pct'] >= 5.0).any()
        
        return is_volume_surge or has_big_rise
    
    def check_trend_condition(self, data):
        """
        检查趋势条件：简化的趋势判断
        :param data: 股票数据
        :return: 是否满足趋势条件
        """
        latest = data.iloc[-1]
        
        # 简化趋势条件：只要5日均线 > 20日均线即可
        ma5 = latest['ma5']
        ma20 = latest['ma20']
        
        if pd.isna(ma5) or pd.isna(ma20):
            return False
            
        return ma5 > ma20
    
    def check_buy_condition(self, data):
        """
        检查买入当天条件
        :param data: 股票数据
        :return: 是否满足买入条件
        """
        latest = data.iloc[-1]
        
        # 简化买入条件：股价接近10日均线（偏离不超过±5%）
        close_price = latest['close']
        ma10 = latest['ma10']
        
        if pd.isna(ma10):
            return False
            
        deviation = abs(close_price - ma10) / ma10
        near_ma10 = deviation <= 0.05  # 放宽到5%
        
        return near_ma10  # 移除阴线要求
    
    def check_sell_condition(self, data, buy_date, current_date, buy_price,
                            highest_price):
        """检查卖出条件"""
        current_data = data.loc[current_date]

        holding_days = len(data.loc[buy_date:current_date]) - 1
        if holding_days >= self.max_holding_days:
            return True, "持仓超期"

        # 次日开盘价低于10日均线
        if holding_days == 1 and current_data['open'] < current_data['ma10']:
            return True, "开盘价低于10日均线"

        # 移动止损
        if highest_price > 0 and current_data['close'] <= highest_price * (1 - self.trail_percent):
            return True, "触发移动止损"

        # 固定止损
        if buy_price > 0 and current_data['close'] <= buy_price * (1 - self.stop_loss_pct):
            return True, "触发止损"

        # 止盈
        if buy_price > 0 and current_data['close'] >= buy_price * (1 + self.profit_take_pct):
            return True, "达到止盈目标"

        # 当日涨幅超过2%
        if current_data['change_pct'] > 2:
            return True, "当日涨幅超过2%"

        return False, ""
    
    def generate_signals(self, stock_code, data):
        """
        生成买卖信号
        :param stock_code: 股票代码
        :param data: 股票数据
        :return: 买卖信号DataFrame
        """
        signals = pd.DataFrame(index=data.index)
        signals['buy'] = False
        signals['sell'] = False
        signals['position'] = 0
        signals['reason'] = ''

        position = 0
        buy_date = None
        buy_price = 0.0
        highest_price = 0.0
        
        for i, (date, row) in enumerate(data.iterrows()):
            if i < 60:  # 降低历史数据要求，只需要60天
                continue
                
            current_data = data.iloc[:i+1]
            
            # 如果没有持仓，检查买入条件
            if position == 0:
                # 检查所有买入条件
                volume_ok = self.check_volume_condition(current_data)
                trend_ok = self.check_trend_condition(current_data)
                buy_ok = self.check_buy_condition(current_data)

                if volume_ok and trend_ok and buy_ok:
                    signals.loc[date, 'buy'] = True
                    signals.loc[date, 'reason'] = '满足买入条件'
                    position = 1
                    buy_date = date
                    buy_price = row['close']
                    highest_price = row['close']
            
            # 如果有持仓，检查卖出条件
            elif position == 1:
                highest_price = max(highest_price, row['close'])
                should_sell, reason = self.check_sell_condition(
                    current_data, buy_date, date, buy_price, highest_price)
                if should_sell:
                    signals.loc[date, 'sell'] = True
                    signals.loc[date, 'reason'] = reason
                    position = 0
                    buy_date = None
                    buy_price = 0.0
                    highest_price = 0.0
            
            signals.loc[date, 'position'] = position
        
        return signals
    
    def backtest_single_stock(self, stock_code, start_date, end_date):
        """
        单只股票回测
        :param stock_code: 股票代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 回测结果
        """
        print(f"正在回测股票: {stock_code}")
        
        # 获取股票数据
        data = self.get_stock_data(stock_code, start_date, end_date)
        if data is None:
            print(f"股票{stock_code}获取数据失败，跳过")
            return None
        
        print(f"股票{stock_code}数据长度: {len(data)}")
        if len(data) < 100:  # 降低数据要求
            print(f"股票{stock_code}数据不足({len(data)}行)，跳过")
            return None
        
        # 生成交易信号
        signals = self.generate_signals(stock_code, data)
        
        # 计算收益
        returns = data['close'].pct_change()
        
        # 使用vectorbt进行回测
        portfolio = vbt.Portfolio.from_signals(
            data['close'],
            signals['buy'],
            signals['sell'],
            init_cash=self.initial_capital * self.position_size,
            fees=0.001  # 手续费0.1%
        )
        
        return {
            'stock_code': stock_code,
            'portfolio': portfolio,
            'data': data,
            'signals': signals,
            'total_return': portfolio.total_return(),
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'max_drawdown': portfolio.max_drawdown(),
            'win_rate': portfolio.trades.win_rate(),
            'total_trades': portfolio.trades.count()
        }
    
    def run_backtest(self, start_date=None, end_date=None, max_stocks=10):
        """
        运行完整回测
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param max_stocks: 最大股票数量
        :return: 回测结果
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"开始回测，时间范围: {start_date} 到 {end_date}")
        
        # 获取股票列表
        stock_list = self.get_hs500_stocks()[:max_stocks]  # 限制股票数量以加快测试
        
        results = []
        for stock_code in stock_list:
            try:
                result = self.backtest_single_stock(stock_code, start_date, end_date)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"回测股票{stock_code}时出错: {e}")
                continue
        
        return results
    
    def analyze_results(self, results):
        """
        分析回测结果
        :param results: 回测结果列表
        """
        if not results:
            print("没有有效的回测结果")
            return
        
        print("\n" + "="*50)
        print("回测结果分析")
        print("="*50)
        
        # 汇总统计
        total_returns = [r['total_return'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results if not pd.isna(r['sharpe_ratio'])]
        max_drawdowns = [r['max_drawdown'] for r in results]
        win_rates = [r['win_rate'] for r in results if not pd.isna(r['win_rate'])]
        total_trades = [r['total_trades'] for r in results]
        
        print(f"回测股票数量: {len(results)}")
        print(f"平均总收益率: {np.mean(total_returns):.2%}")
        print(f"收益率标准差: {np.std(total_returns):.2%}")
        print(f"最大收益率: {np.max(total_returns):.2%}")
        print(f"最小收益率: {np.min(total_returns):.2%}")
        
        if sharpe_ratios:
            print(f"平均夏普比率: {np.mean(sharpe_ratios):.2f}")
        
        print(f"平均最大回撤: {np.mean(max_drawdowns):.2%}")
        
        if win_rates:
            print(f"平均胜率: {np.mean(win_rates):.2%}")
        
        print(f"平均交易次数: {np.mean(total_trades):.1f}")
        
        # 详细结果
        print("\n详细结果:")
        print("-"*80)
        print(f"{'股票代码':<10} {'总收益率':<10} {'夏普比率':<10} {'最大回撤':<10} {'胜率':<10} {'交易次数':<10}")
        print("-"*80)
        
        for result in results:
            stock_code = result['stock_code']
            total_return = result['total_return']
            sharpe_ratio = result['sharpe_ratio'] if not pd.isna(result['sharpe_ratio']) else 0
            max_drawdown = result['max_drawdown']
            win_rate = result['win_rate'] if not pd.isna(result['win_rate']) else 0
            trades = result['total_trades']
            
            print(f"{stock_code:<10} {total_return:<10.2%} {sharpe_ratio:<10.2f} {max_drawdown:<10.2%} {win_rate:<10.2%} {trades:<10}")
        
        # 绘制结果图表
        self.plot_results(results)
    
    def plot_results(self, results):
        """
        绘制回测结果图表
        :param results: 回测结果列表
        """
        if not results:
            return
        
        # 创建综合图表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 整体净值曲线
        ax1 = plt.subplot(3, 2, 1)
        self.plot_equity_curve(results, ax1)
        
        # 2. 收益率分析
        ax2 = plt.subplot(3, 2, 2)
        self.plot_returns_analysis(results, ax2)
        
        # 3. 最大回撤分析（卡码率）
        ax3 = plt.subplot(3, 2, 3)
        self.plot_drawdown_analysis(results, ax3)
        
        # 4. 夏普比率分析
        ax4 = plt.subplot(3, 2, 4)
        self.plot_sharpe_analysis(results, ax4)
        
        # 5. 策略表现汇总
        ax5 = plt.subplot(3, 2, 5)
        self.plot_performance_summary(results, ax5)
        
        # 6. 月度收益热力图
        ax6 = plt.subplot(3, 2, 6)
        self.plot_monthly_returns(results, ax6)
        
        plt.suptitle('量价策略回测结果综合分析', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig('strategy_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_equity_curve(self, results, ax):
        """绘制整体净值曲线"""
        # 合并所有股票的净值曲线
        all_equity_curves = []
        
        for result in results:
            portfolio = result['portfolio']
            equity_curve = portfolio.value()
            equity_curve.name = result['stock_code']
            all_equity_curves.append(equity_curve)
        
        if all_equity_curves:
            # 计算等权重组合净值
            combined_equity = pd.concat(all_equity_curves, axis=1).fillna(method='ffill').mean(axis=1)
            
            # 绘制净值曲线
            ax.plot(combined_equity.index, combined_equity.values, linewidth=2, color='blue', label='策略净值')
            
            # 计算基准（假设为0收益率）
            benchmark = pd.Series(self.initial_capital * self.position_size, 
                                index=combined_equity.index, name='基准')
            ax.plot(benchmark.index, benchmark.values, linewidth=1, color='red', 
                   linestyle='--', alpha=0.7, label='基准（无收益）')
            
            ax.set_title('策略净值曲线', fontsize=14, fontweight='bold')
            ax.set_xlabel('日期')
            ax.set_ylabel('净值')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加收益率标注
            total_return = (combined_equity.iloc[-1] / combined_equity.iloc[0] - 1) * 100
            ax.text(0.02, 0.98, f'总收益率: {total_return:.2f}%', 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top')
    
    def plot_returns_analysis(self, results, ax):
        """绘制收益率分析"""
        returns = [r['total_return'] * 100 for r in results]  # 转换为百分比
        
        # 绘制收益率分布直方图
        n, bins, patches = ax.hist(returns, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 根据收益率给直方图着色
        for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
            if bin_val > 0:
                patch.set_facecolor('lightgreen')
            else:
                patch.set_facecolor('lightcoral')
        
        # 添加统计线
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_return:.2f}%')
        ax.axvline(median_return, color='orange', linestyle='--', linewidth=2, label=f'中位数: {median_return:.2f}%')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='盈亏平衡线')
        
        ax.set_title('收益率分布分析', fontsize=14, fontweight='bold')
        ax.set_xlabel('收益率 (%)')
        ax.set_ylabel('股票数量')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
        ax.text(0.02, 0.98, f'胜率: {win_rate:.1f}%\n标准差: {np.std(returns):.2f}%', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top')
    
    def plot_drawdown_analysis(self, results, ax):
        """绘制最大回撤分析（卡码率）"""
        drawdowns = [r['max_drawdown'] * 100 for r in results]  # 转换为百分比
        
        # 绘制回撤分布
        ax.hist(drawdowns, bins=15, alpha=0.7, color='orange', edgecolor='black')
        
        # 添加统计线
        mean_drawdown = np.mean(drawdowns)
        max_drawdown = np.max(drawdowns)
        ax.axvline(mean_drawdown, color='red', linestyle='--', linewidth=2, 
                  label=f'平均回撤: {mean_drawdown:.2f}%')
        ax.axvline(max_drawdown, color='darkred', linestyle='--', linewidth=2, 
                  label=f'最大回撤: {max_drawdown:.2f}%')
        
        ax.set_title('最大回撤分布分析（卡码率）', fontsize=14, fontweight='bold')
        ax.set_xlabel('最大回撤 (%)')
        ax.set_ylabel('股票数量')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加风险评估
        risk_level = "低" if mean_drawdown < 5 else "中" if mean_drawdown < 10 else "高"
        ax.text(0.02, 0.98, f'风险等级: {risk_level}\n回撤标准差: {np.std(drawdowns):.2f}%', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top')
    
    def plot_sharpe_analysis(self, results, ax):
        """绘制夏普比率分析"""
        sharpe_ratios = [r['sharpe_ratio'] for r in results 
                        if not pd.isna(r['sharpe_ratio']) and np.isfinite(r['sharpe_ratio'])]
        
        if sharpe_ratios:
            # 绘制夏普比率分布
            n, bins, patches = ax.hist(sharpe_ratios, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            
            # 根据夏普比率给直方图着色
            for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
                if bin_val > 1:
                    patch.set_facecolor('darkgreen')
                elif bin_val > 0:
                    patch.set_facecolor('lightgreen')
                else:
                    patch.set_facecolor('lightcoral')
            
            # 添加统计线
            mean_sharpe = np.mean(sharpe_ratios)
            ax.axvline(mean_sharpe, color='red', linestyle='--', linewidth=2, 
                      label=f'平均夏普: {mean_sharpe:.2f}')
            ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='零线')
            ax.axvline(1, color='green', linestyle=':', alpha=0.7, label='优秀线(1.0)')
            
            ax.set_title('夏普比率分布分析', fontsize=14, fontweight='bold')
            ax.set_xlabel('夏普比率')
            ax.set_ylabel('股票数量')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加评级
            excellent_count = len([s for s in sharpe_ratios if s > 1])
            good_count = len([s for s in sharpe_ratios if 0 < s <= 1])
            poor_count = len([s for s in sharpe_ratios if s <= 0])
            
            ax.text(0.02, 0.98, f'优秀(>1): {excellent_count}只\n良好(0-1): {good_count}只\n较差(≤0): {poor_count}只', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top')
        else:
            ax.text(0.5, 0.5, '无有效夏普比率数据', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('夏普比率分布分析', fontsize=14, fontweight='bold')
    
    def plot_performance_summary(self, results, ax):
        """绘制策略表现汇总"""
        # 准备数据
        metrics = ['总收益率', '夏普比率', '最大回撤', '胜率', '交易次数']
        
        returns = [r['total_return'] * 100 for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results 
                        if not pd.isna(r['sharpe_ratio']) and np.isfinite(r['sharpe_ratio'])]
        drawdowns = [r['max_drawdown'] * 100 for r in results]
        win_rates = [r['win_rate'] * 100 for r in results if not pd.isna(r['win_rate'])]
        trade_counts = [r['total_trades'] for r in results]
        
        values = [
            np.mean(returns),
            np.mean(sharpe_ratios) if sharpe_ratios else 0,
            np.mean(drawdowns),
            np.mean(win_rates) if win_rates else 0,
            np.mean(trade_counts)
        ]
        
        # 创建条形图
        colors = ['green' if v > 0 else 'red' for v in values[:2]] + ['orange', 'blue', 'purple']
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('策略关键指标汇总', fontsize=14, fontweight='bold')
        ax.set_ylabel('数值')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    def plot_monthly_returns(self, results, ax):
        """绘制月度收益热力图"""
        if not results:
            return
        
        # 收集所有月度收益数据
        monthly_returns_data = []
        
        for result in results:
            portfolio = result['portfolio']
            returns = portfolio.returns()
            
            if len(returns) > 0:
                # 按月分组计算收益
                monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns_data.append(monthly_returns)
        
        if monthly_returns_data:
            # 合并所有月度收益
            combined_monthly = pd.concat(monthly_returns_data, axis=1).mean(axis=1) * 100
            
            # 创建年月矩阵
            years = combined_monthly.index.year.unique()
            months = range(1, 13)
            
            heatmap_data = np.full((len(years), 12), np.nan)
            
            for i, year in enumerate(years):
                year_data = combined_monthly[combined_monthly.index.year == year]
                for month_return in year_data.items():
                    month = month_return[0].month
                    heatmap_data[i, month-1] = month_return[1]
            
            # 绘制热力图
            im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
            
            # 设置标签
            ax.set_xticks(range(12))
            ax.set_xticklabels(['1月', '2月', '3月', '4月', '5月', '6月',
                               '7月', '8月', '9月', '10月', '11月', '12月'])
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(years)
            
            # 添加数值标签
            for i in range(len(years)):
                for j in range(12):
                    if not np.isnan(heatmap_data[i, j]):
                        text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                                     ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title('月度收益热力图', fontsize=14, fontweight='bold')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('月度收益率 (%)', rotation=270, labelpad=15)
        else:
            ax.text(0.5, 0.5, '无足够数据绘制月度收益', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('月度收益热力图', fontsize=14, fontweight='bold')


def main():
    """主函数"""
    print("股票量价策略回测系统")
    print("="*50)
    
    # 创建策略实例
    strategy = VolumePrice_Strategy(initial_capital=1000000)
    
    # 设置回测时间范围（最近一年）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"回测时间范围: {start_date} 到 {end_date}")
    
    try:
        # 运行回测（限制股票数量以加快测试）
        results = strategy.run_backtest(start_date=start_date, end_date=end_date, max_stocks=20)
        
        # 分析结果
        strategy.analyze_results(results)
        
        print("\n回测完成！结果图表已保存为 'strategy_backtest_results.png'")
        
    except Exception as e:
        print(f"回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
