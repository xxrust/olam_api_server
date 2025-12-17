from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
from dotenv import load_dotenv
import os
import logging
import flask
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 数据库配置
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# 打印数据库配置（不包含密码）
logger.info(f"数据库配置: {DB_CONFIG['dbname']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")

app = Flask(__name__)
CORS(app)  # 启用 CORS

# 添加根路由
@app.route('/')
def index():
    return "API 服务器正在运行"

# 添加一个简单的测试路由
@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({"status": "success", "message": "服务器正在运行"})

def get_db_connection():
    """获取数据库连接"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("数据库连接成功")
        return conn
    except Exception as e:
        logger.error(f"数据库连接失败: {str(e)}")
        raise

def execute_query(query: str, params: tuple = None):
    """执行查询并返回结果"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                logger.info(f"执行查询: {query}")
                logger.info(f"参数: {params}")
                cur.execute(query, params or ())
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    result = [dict(zip(columns, row)) for row in cur.fetchall()]
                    logger.info(f"查询结果: {result}")
                    return result
                return None
    except Exception as e:
        logger.error(f"查询执行失败: {str(e)}")
        raise

def execute_single_query(query: str, params: tuple = None):
    """执行查询并返回单条结果"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                logger.info(f"执行查询: {query}")
                logger.info(f"参数: {params}")
                cur.execute(query, params or ())
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    row = cur.fetchone()
                    result = dict(zip(columns, row)) if row else None
                    logger.info(f"查询结果: {result}")
                    return result
                return None
    except Exception as e:
        logger.error(f"查询执行失败: {str(e)}")
        raise

def get_all_tables():
    """获取数据库中的所有表"""
    try:
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                tables = [row[0] for row in cur.fetchall()]
                logger.info(f"数据库中的表: {tables}")
                return tables
    except Exception as e:
        logger.error(f"获取数据库表失败: {str(e)}")
        return []

def get_table_columns(table_name):
    """获取表的所有列"""
    try:
        query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position;
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (table_name,))
                columns = [(row[0], row[1]) for row in cur.fetchall()]
                logger.info(f"表 {table_name} 的列: {columns}")
                return columns
    except Exception as e:
        logger.error(f"获取表列失败: {str(e)}")
        return []

@app.route('/api/db-info', methods=['GET'])
def get_db_info():
    """获取数据库信息"""
    try:
        tables = get_all_tables()
        tables_info = {}
        
        for table in tables:
            tables_info[table] = get_table_columns(table)
            
        return jsonify({
            "status": "success",
            "message": "数据库信息获取成功",
            "tables": tables,
            "tables_info": {table: [{"name": col[0], "type": col[1]} for col in cols] for table, cols in tables_info.items()}
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/test-db', methods=['GET'])
def test_db_connection():
    """测试数据库连接"""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()
            
            # 获取所有表
            tables = get_all_tables()
            
        return jsonify({
            "status": "success",
            "message": "数据库连接成功",
            "version": version[0],
            "tables": tables
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/batches/count', methods=['GET'])
def get_batch_count():
    """获取总批次数量"""
    try:
        # 使用 olam 表作为主表
        query = """
            SELECT COUNT(*) as count 
            FROM olam
        """
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                count = cur.fetchone()[0]
                
        return jsonify({"count": count})
    except Exception as e:
        logger.error(f"获取批次数量失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batches', methods=['GET'])
def get_batches():
    """获取批次数据列表"""
    try:
        # 获取查询参数
        batch_ids = request.args.getlist('batchIds')  # 获取批次ID参数
        device_ids = request.args.getlist('deviceIds')
        operator_ids = request.args.getlist('operatorIds')
        lot_numbers = request.args.getlist('lotNumbers')  # 获取Lot号参数
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        target_f_min = request.args.get('targetFMin', type=float)
        target_f_max = request.args.get('targetFMax', type=float)
        limit = request.args.get('limit', 100, type=int)  # 默认100，但前端通常会传5
        
        # 限制最大值和最小值，防止查询过大
        limit = max(1, min(limit, 100))
        
        logger.info(f"获取批次数据，限制为 {limit} 条")
        
        # 构建WHERE条件
        conditions = []
        params = []
        
        # 批次ID条件 - 优先级最高
        if batch_ids:
            batch_placeholders = ', '.join(['%s'] * len(batch_ids))
            conditions.append(f"o.batch_id IN ({batch_placeholders})")
            params.extend(batch_ids)
        # Lot号条件 - 次优先级
        elif lot_numbers:
            lot_placeholders = ', '.join(['%s'] * len(lot_numbers))
            conditions.append(f"o.lot IN ({lot_placeholders})")
            params.extend(lot_numbers)
        # 只有在没有指定批次ID和Lot号时，才应用其他筛选条件
        else:
            # 多个设备ID条件
            if device_ids:
                device_placeholders = ', '.join(['%s'] * len(device_ids))
                conditions.append(f"o.device_id IN ({device_placeholders})")
                params.extend(device_ids)
            
            # 多个操作员ID条件
            if operator_ids:
                operator_placeholders = ', '.join(['%s'] * len(operator_ids))
                conditions.append(f"o.employee_id IN ({operator_placeholders})")
                params.extend(operator_ids)
            
            # 日期范围条件
            if start_date:
                conditions.append("DATE(o.start_time) >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("DATE(o.start_time) <= %s")
                params.append(end_date)
            
            # 频率范围条件
            if target_f_min is not None:
                conditions.append("o.target_f >= %s")
                params.append(target_f_min)
            if target_f_max is not None:
                conditions.append("o.target_f <= %s")
                params.append(target_f_max)
        
        # 构建WHERE子句
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # 构建查询 - 直接从olam表中查询主要数据，不使用复杂子查询
        query = f"""
            SELECT 
                o.batch_id,
                o.lot,
                o.device_id,
                o.employee_id as operator_id,
                o.start_time,
                o.end_time,
                o.start_f,
                o.target_f,
                o.final_f,
                0 as stdev
            FROM olam o
            WHERE {where_clause}
            ORDER BY o.start_time DESC
            LIMIT {limit}
        """
        
        # 执行查询
        batches = execute_query(query, params)
        
        # 如果获取到批次，但需要附加标准差数据
        if batches:
            # 获取所有批次ID
            batch_ids = [batch['batch_id'] for batch in batches]
            
            # 避免SQL注入，使用参数化查询
            batch_placeholders = ', '.join(['%s'] * len(batch_ids))
            
            # 单独查询这些批次的标准差数据
            stdev_query = f"""
                SELECT 
                    batch_id,
                    STDDEV(fre) as stdev
                FROM last_round_f
                WHERE batch_id IN ({batch_placeholders})
                GROUP BY batch_id
            """
            
            stdev_data = {}
            try:
                stdev_results = execute_query(stdev_query, batch_ids)
                if stdev_results:
                    # 建立批次ID到标准差的映射
                    for row in stdev_results:
                        stdev_data[row['batch_id']] = row['stdev']
                    
                    # 更新批次数据中的标准差
                    for batch in batches:
                        batch_id = batch['batch_id']
                        if batch_id in stdev_data:
                            batch['stdev'] = float(stdev_data[batch_id])
                        else:
                            batch['stdev'] = 0.0
            except Exception as e:
                logger.warning(f"获取标准差数据失败: {str(e)}")
                # 继续处理，不中断主流程
                for batch in batches:
                    batch['stdev'] = 0.0
        
        # 返回结果
        return jsonify(batches)
        
    except Exception as e:
        logger.error(f"获取批次数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """获取设备列表"""
    try:
        # 从 olam 表中提取唯一设备
        query = """
            SELECT DISTINCT device_id as device_id, device_id as device_name
            FROM olam
            ORDER BY device_id
        """
        
        result = execute_query(query)
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取设备列表失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/operators', methods=['GET'])
def get_operators():
    """获取操作员列表"""
    try:
        # 从 olam 表中提取唯一员工ID
        query = """
            SELECT DISTINCT employee_id as operator_id, employee_id as operator_name
            FROM olam
            ORDER BY employee_id
        """
        
        result = execute_query(query)
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取操作员列表失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/grid', methods=['GET'])
def get_grid_records():
    """获取 grid(修盘) 表数据列表。"""
    try:
        device_id = request.args.get('deviceId')
        grid_mod_raw = request.args.get('gridMod') or request.args.get('repairMethod')
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        limit = request.args.get('limit', 200, type=int)
        offset = request.args.get('offset', 0, type=int)

        limit = max(1, min(limit, 1000))
        offset = max(0, offset)

        grid_mod = None
        if grid_mod_raw not in (None, ''):
            try:
                grid_mod = int(grid_mod_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "gridMod 必须为整数"}), 400

        conditions = []
        params = []

        if device_id:
            conditions.append("g.device_id = %s")
            params.append(device_id)
        if grid_mod is not None:
            conditions.append("g.grid_mod = %s")
            params.append(grid_mod)
        if start_date:
            conditions.append("DATE(g.start_time) >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("DATE(g.start_time) <= %s")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT
                g.id,
                g.device_id,
                g.grid_mod,
                g.set_round,
                g.start_time,
                g.end_time,
                g.round_num,
                g.end_way,
                CASE
                    WHEN g.start_time IS NOT NULL AND g.end_time IS NOT NULL
                    THEN EXTRACT(EPOCH FROM (g.end_time - g.start_time))
                    ELSE NULL
                END AS duration_seconds
            FROM grid g
            WHERE {where_clause}
            ORDER BY g.start_time DESC NULLS LAST, g.id DESC
            LIMIT {limit} OFFSET {offset}
        """

        result = execute_query(query, tuple(params))
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取grid(修盘)数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/grid/options', methods=['GET'])
def get_grid_options():
    """获取 grid_mod/end_way 的可选值（用于前端下拉）。"""
    try:
        grid_mods = execute_query("""
            SELECT DISTINCT grid_mod
            FROM grid
            WHERE grid_mod IS NOT NULL
            ORDER BY grid_mod
        """) or []
        end_ways = execute_query("""
            SELECT DISTINCT end_way
            FROM grid
            WHERE end_way IS NOT NULL
            ORDER BY end_way
        """) or []

        return jsonify({
            "gridMods": [row["grid_mod"] for row in grid_mods],
            "endWays": [row["end_way"] for row in end_ways],
        })
    except Exception as e:
        logger.error(f"获取grid options失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batches/sample', methods=['GET'])
def get_batch_sample():
    """获取批次数据样本"""
    try:
        query = """
            SELECT 
                batch_id,
                created_at,
                resonant,
                stdev
            FROM batches
            LIMIT 5
        """
        result = execute_query(query)
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取批次样本失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analysis/initial-variance', methods=['GET'])
def analyze_initial_variance():
    """分析批次前几圈的散差情况"""
    try:
        batch_id = request.args.get('batchId')
        round_count = request.args.get('roundCount', 3, type=int)
        
        if not batch_id:
            return jsonify({"error": "缺少批次ID参数"}), 400
            
        # 查询该批次前几圈的实时数据
        query = """
            SELECT 
                resonant, 
                stdev, 
                count, 
                created_at
            FROM realtime_freq
            WHERE batch_id = %s AND count <= %s
            ORDER BY count, created_at
        """
        
        data = execute_query(query, (batch_id, round_count))
        
        if not data:
            return jsonify({"error": f"批次 {batch_id} 未找到数据"}), 404
        
        # 按圈分组数据
        round_data = {}
        for row in data:
            count = row['count']
            if count not in round_data:
                round_data[count] = []
            round_data[count].append({
                'resonant': row['resonant'],
                'stdev': row['stdev'],
                'time': row['created_at']
            })
        
        # 分析每圈的数据
        analysis = []
        for count in sorted(round_data.keys()):
            points = round_data[count]
            if points:
                last_point = points[-1]
                first_point = points[0]
                
                # 计算时间差（以秒为单位）
                time_span = (last_point['time'] - first_point['time']).total_seconds() if hasattr(last_point['time'], 'total_seconds') else 0
                
                analysis.append({
                    "round": count,
                    "resonant": last_point['resonant'],
                    "stdev": last_point['stdev'],
                    "time_span": time_span,
                    "data_points": len(points)
                })
        
        return jsonify({
            "round_analysis": analysis,
            "total_rounds": len(round_data),
            "analyzed_rounds": len(analysis)
        })
        
    except Exception as e:
        logger.error(f"分析初始散差失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analysis/repair-effect', methods=['GET'])
def analyze_repair_effect():
    """分析修盘后不同批次的质量变化"""
    try:
        grid_id = request.args.get('gridId', type=int)
        device_id = request.args.get('deviceId')
        grid_mod = request.args.get('repairMethod')
        max_batches = request.args.get('maxBatches', 10, type=int)
        
        max_batches = max(1, min(max_batches, 200))

        # 优先：按单次修盘记录分析（选中某条修盘记录后，分析其后第 N 盘，直到下一次修盘为止）
        if grid_id is not None:
            query = """
                WITH g AS (
                    SELECT
                        id,
                        device_id,
                        grid_mod,
                        COALESCE(end_time, start_time) as grid_time
                    FROM grid
                    WHERE id = %s
                ),
                next_g AS (
                    SELECT COALESCE(start_time, end_time) as next_grid_time
                    FROM grid
                    WHERE device_id = (SELECT device_id FROM g)
                      AND COALESCE(start_time, end_time) IS NOT NULL
                      AND COALESCE(start_time, end_time) > (SELECT grid_time FROM g)
                    ORDER BY COALESCE(start_time, end_time), id
                    LIMIT 1
                ),
                candidate_batches AS (
                    SELECT
                        o.batch_id,
                        o.start_time
                    FROM olam o
                    WHERE o.device_id = (SELECT device_id FROM g)
                      AND o.start_time >= (SELECT grid_time FROM g)
                      AND (
                           (SELECT next_grid_time FROM next_g) IS NULL
                           OR o.start_time < (SELECT next_grid_time FROM next_g)
                      )
                    ORDER BY o.start_time, o.batch_id
                    LIMIT %s
                ),
                ranked AS (
                    SELECT
                        cb.batch_id,
                        cb.start_time,
                        s.stdev,
                        ROW_NUMBER() OVER (ORDER BY start_time, batch_id) as batch_number
                    FROM candidate_batches cb
                    LEFT JOIN (
                        SELECT
                            batch_id,
                            STDDEV(fre)::double precision as stdev
                        FROM last_round_f
                        WHERE batch_id IN (SELECT batch_id FROM candidate_batches)
                        GROUP BY batch_id
                    ) s ON s.batch_id = cb.batch_id
                )
                SELECT
                    batch_number,
                    AVG(stdev)::double precision as avg_stdev,
                    NULL::double precision as std_stdev,
                    COUNT(*)::int as batch_count
                FROM ranked
                GROUP BY batch_number
                ORDER BY batch_number
            """

            result = execute_query(query, (grid_id, max_batches))
            meta = execute_single_query("""
                SELECT
                    g.id as grid_id,
                    g.device_id,
                    g.grid_mod,
                    COALESCE(g.end_time, g.start_time) as grid_time,
                    (
                        SELECT COALESCE(g2.start_time, g2.end_time)
                        FROM grid g2
                        WHERE g2.device_id = g.device_id
                          AND COALESCE(g2.start_time, g2.end_time) IS NOT NULL
                          AND COALESCE(g2.start_time, g2.end_time) > COALESCE(g.end_time, g.start_time)
                        ORDER BY COALESCE(g2.start_time, g2.end_time), g2.id
                        LIMIT 1
                    ) as next_grid_time
                FROM grid g
                WHERE g.id = %s
            """, (grid_id,))

            debug = execute_single_query("""
                WITH g AS (
                    SELECT
                        id,
                        device_id,
                        COALESCE(end_time, start_time) as grid_time
                    FROM grid
                    WHERE id = %s
                ),
                next_g AS (
                    SELECT COALESCE(start_time, end_time) as next_grid_time
                    FROM grid
                    WHERE device_id = (SELECT device_id FROM g)
                      AND COALESCE(start_time, end_time) IS NOT NULL
                      AND COALESCE(start_time, end_time) > (SELECT grid_time FROM g)
                    ORDER BY COALESCE(start_time, end_time), id
                    LIMIT 1
                ),
                candidates AS (
                    SELECT o.batch_id
                    FROM olam o
                    WHERE o.device_id = (SELECT device_id FROM g)
                      AND o.start_time >= (SELECT grid_time FROM g)
                      AND (
                           (SELECT next_grid_time FROM next_g) IS NULL
                           OR o.start_time < (SELECT next_grid_time FROM next_g)
                      )
                    ORDER BY o.start_time, o.batch_id
                    LIMIT %s
                )
                SELECT
                    (SELECT COUNT(*) FROM candidates)::int as candidate_batches
            """, (grid_id, max_batches))

            return jsonify({"meta": meta, "debug": debug, "batch_analysis": result})

        # 兼容：按设备 + 修盘方式聚合分析（跨多次修盘取平均）
        if not device_id or not grid_mod:
            return jsonify({"error": "缺少设备ID或修盘方式参数"}), 400
        
        # 查询修盘后的批次质量数据（按修盘后的“研磨盘数/批次数”分析）
        # 逻辑：以每次修盘时间为起点，找该设备后续的批次；截至下一次修盘（任意方式）为止，
        # 对每次修盘后的第 N 盘（N=1,2,3...）聚合 stdev。
        query = """
            WITH grids AS (
                SELECT 
                    g.id,
                    g.device_id,
                    g.grid_mod,
                    COALESCE(g.end_time, g.start_time) as grid_time,
                    LEAD(COALESCE(g.start_time, g.end_time)) OVER (
                        PARTITION BY g.device_id
                        ORDER BY COALESCE(g.start_time, g.end_time), g.id
                    ) as next_grid_time
                FROM grid g
                WHERE g.device_id = %s
                AND g.grid_mod = %s
                AND COALESCE(g.end_time, g.start_time) IS NOT NULL
            ),
            batches_with_stdev AS (
                SELECT 
                    o.batch_id,
                    o.device_id,
                    o.start_time,
                    lrf.stdev
                FROM olam o
                JOIN (
                    SELECT 
                        batch_id, 
                        STDDEV(fre) as stdev
                    FROM last_round_f
                    GROUP BY batch_id
                ) lrf ON lrf.batch_id = o.batch_id
                WHERE o.device_id = %s
            ),
            grid_batches AS (
                SELECT 
                    g.id as grid_id,
                    g.grid_time,
                    b.batch_id,
                    b.start_time,
                    b.stdev,
                    ROW_NUMBER() OVER (
                        PARTITION BY g.id 
                        ORDER BY b.start_time, b.batch_id
                    ) as batch_number
                FROM grids g
                JOIN batches_with_stdev b
                  ON b.start_time > g.grid_time
                 AND (
                      g.next_grid_time IS NULL
                      OR b.start_time <= g.next_grid_time
                 )
            )
            SELECT 
                batch_number,
                AVG(stdev)::double precision as avg_stdev,
                STDDEV(stdev)::double precision as std_stdev,
                COUNT(*)::int as batch_count
            FROM grid_batches
            WHERE batch_number <= %s
            GROUP BY batch_number
            ORDER BY batch_number
        """
        
        result = execute_query(query, (device_id, grid_mod, device_id, max_batches))
        return jsonify({"batch_analysis": result})
        
    except Exception as e:
        logger.error(f"分析修盘效果失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analysis/frequency-range', methods=['POST'])
def analyze_frequency_range():
    """按不同频率范围分析散差"""
    try:
        data = request.json
        device_id = data.get('deviceId')
        frequency_ranges = data.get('frequencyRanges')
        
        if not device_id or not frequency_ranges:
            return jsonify({"error": "缺少设备ID或频率范围参数"}), 400
        
        results = {}
        for freq_range in frequency_ranges:
            min_freq = freq_range.get('min')
            max_freq = freq_range.get('max')
            
            if min_freq is None or max_freq is None:
                continue
            
            # 查询特定频率范围的批次散差数据
            query = """
                SELECT 
                    AVG(lrf.stdev) as avg_stdev,
                    STDDEV(lrf.stdev) as std_stdev,
                    COUNT(*) as batch_count
                FROM olam o
                JOIN (
                    SELECT 
                        batch_id, 
                        STDDEV(fre) as stdev
                    FROM last_round_f
                    GROUP BY batch_id
                ) lrf ON o.batch_id = lrf.batch_id
                WHERE o.device_id = %s
                AND o.target_f BETWEEN %s AND %s
            """
            
            result = execute_single_query(query, (device_id, min_freq, max_freq))
            
            if result:
                range_key = f"{min_freq}-{max_freq}"
                results[range_key] = result
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"分析频率范围失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analysis/operator-device-impact', methods=['GET'])
def analyze_operator_device_impact():
    """分析操作员和设备对散差的影响"""
    try:
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        
        if not start_date or not end_date:
            return jsonify({"error": "缺少开始日期或结束日期参数"}), 400
        
        # 分析操作员影响
        operator_query = """
            SELECT 
                o.employee_id,
                AVG(lrf.stdev) as avg_stdev,
                STDDEV(lrf.stdev) as std_stdev,
                COUNT(*) as batch_count
            FROM olam o
            JOIN (
                SELECT 
                    batch_id, 
                    STDDEV(fre) as stdev
                FROM last_round_f
                GROUP BY batch_id
            ) lrf ON o.batch_id = lrf.batch_id
            WHERE o.start_time BETWEEN %s AND %s
            GROUP BY o.employee_id
            ORDER BY avg_stdev ASC
        """
        
        operator_results = execute_query(operator_query, (start_date, end_date))
        
        # 分析设备影响
        device_query = """
            SELECT 
                o.device_id,
                AVG(lrf.stdev) as avg_stdev,
                STDDEV(lrf.stdev) as std_stdev,
                COUNT(*) as batch_count
            FROM olam o
            JOIN (
                SELECT 
                    batch_id, 
                    STDDEV(fre) as stdev
                FROM last_round_f
                GROUP BY batch_id
            ) lrf ON o.batch_id = lrf.batch_id
            WHERE o.start_time BETWEEN %s AND %s
            GROUP BY o.device_id
            ORDER BY avg_stdev ASC
        """
        
        device_results = execute_query(device_query, (start_date, end_date))
        
        return jsonify({
            "operator_impact": operator_results,
            "device_impact": device_results
        })
        
    except Exception as e:
        logger.error(f"分析操作员和设备影响失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batches/rounds', methods=['GET'])
def get_batch_rounds():
    """获取批次的每圈频率数据"""
    try:
        # 获取查询参数
        batch_ids = request.args.getlist('batchIds')
        
        if not batch_ids:
            return jsonify({"error": "缺少批次ID参数"}), 400
            
        # 将批次ID参数化，避免SQL注入
        placeholders = ', '.join(['%s'] * len(batch_ids))
        
        # 一次性查询所有批次的数据
        query = f"""
            WITH ranked_data AS (
                SELECT 
                    batch_id,
                    count,
                    resonant,
                    stdev,
                    ROW_NUMBER() OVER(PARTITION BY batch_id, count ORDER BY id DESC) as rn
                FROM realtime_freq
                WHERE batch_id IN ({placeholders})
            )
            SELECT 
                batch_id,
                count,
                resonant,
                stdev
            FROM ranked_data
            WHERE rn = 1
            ORDER BY batch_id, count
        """
        
        logger.info(f"执行查询: {query}")
        all_data = execute_query(query, batch_ids)
        
        # 将查询结果按批次ID分组
        results = {}
        for row in all_data:
            batch_id = row['batch_id']
            if batch_id not in results:
                results[batch_id] = []
                
            results[batch_id].append({
                'count': row['count'],
                'resonant': row['resonant'],
                'stdev': row['stdev'],
                'time': None
            })
        
        # 确保每个批次的数据按圈数排序
        for batch_id in results:
            results[batch_id].sort(key=lambda x: x['count'])
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"获取批次圈数数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/olam', methods=['GET'])
def get_olam_data():
    """获取olam表数据（带分页）"""
    try:
        # 获取查询参数
        batch_ids = request.args.getlist('batchIds')  # 获取批次ID参数
        device_ids = request.args.getlist('deviceIds')
        operator_ids = request.args.getlist('operatorIds')
        lot_numbers = request.args.getlist('lotNumbers')
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        target_f_min = request.args.get('targetFMin', type=float)
        target_f_max = request.args.get('targetFMax', type=float)
        
        # 分页参数
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('pageSize', 20, type=int)
        
        # 限制最大页面大小
        page_size = min(page_size, 100)
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建WHERE条件
        conditions = []
        params = []
        
        # 批次ID条件 - 优先级最高
        if batch_ids:
            batch_placeholders = ', '.join(['%s'] * len(batch_ids))
            conditions.append(f"o.batch_id IN ({batch_placeholders})")
            params.extend(batch_ids)
        # 只有在没有指定批次ID时，才应用其他筛选条件
        else:
            # 多个设备ID条件
            if device_ids:
                device_placeholders = ', '.join(['%s'] * len(device_ids))
                conditions.append(f"o.device_id IN ({device_placeholders})")
                params.extend(device_ids)
            
            # 多个操作员ID条件
            if operator_ids:
                operator_placeholders = ', '.join(['%s'] * len(operator_ids))
                conditions.append(f"o.employee_id IN ({operator_placeholders})")
                params.extend(operator_ids)
                
            # 处理Lot号筛选
            if lot_numbers:
                lot_placeholders = ', '.join(['%s'] * len(lot_numbers))
                conditions.append(f"o.lot IN ({lot_placeholders})")
                params.extend(lot_numbers)
                
            # 日期范围条件
            if start_date:
                conditions.append("DATE(o.start_time) >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("DATE(o.start_time) <= %s")
                params.append(end_date)
            
            # 频率范围条件
            if target_f_min is not None:
                conditions.append("o.target_f >= %s")
                params.append(target_f_min)
            if target_f_max is not None:
                conditions.append("o.target_f <= %s")
                params.append(target_f_max)
        
        # 构建WHERE子句
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # 先查询满足条件的总记录数
        count_query = f"""
            SELECT COUNT(*) as total
            FROM olam o
            WHERE {where_clause}
        """
        
        count_result = execute_single_query(count_query, params)
        total_count = count_result.get('total', 0) if count_result else 0
        
        # 查询分页数据 - 先不查标准差，提高主查询速度
        data_query = f"""
            SELECT 
                o.batch_id,
                o.lot,
                o.device_id,
                o.employee_id as operator_id,
                o.start_time,
                o.end_time,
                o.start_f,
                o.target_f,
                o.final_f
            FROM olam o
            WHERE {where_clause}
            ORDER BY o.start_time DESC
            LIMIT {page_size} OFFSET {offset}
        """
        
        items = execute_query(data_query, params)
        
        # 如果有数据，再单独查询标准差
        if items:
            # 获取所有批次ID
            batch_ids = [item['batch_id'] for item in items]
            
            # 避免SQL注入，使用参数化查询
            batch_placeholders = ', '.join(['%s'] * len(batch_ids))
            
            # 单独查询这些批次的标准差数据
            stdev_query = f"""
                SELECT 
                    batch_id,
                    STDDEV(fre) as stdev
                FROM last_round_f
                WHERE batch_id IN ({batch_placeholders})
                GROUP BY batch_id
            """
            
            stdev_data = {}
            try:
                stdev_results = execute_query(stdev_query, batch_ids)
                if stdev_results:
                    # 创建批次ID到标准差的映射
                    for row in stdev_results:
                        stdev_data[row['batch_id']] = row['stdev']
                    
                    # 更新批次数据
                    for item in items:
                        batch_id = item['batch_id']
                        item['stdev'] = float(stdev_data.get(batch_id, 0))
                        
            except Exception as e:
                logger.warning(f"获取标准差数据失败: {str(e)}")
                # 出错时也要继续处理，为所有项目设置默认值
                for item in items:
                    item['stdev'] = 0.0
        
        # 返回结果，包含分页信息
        return jsonify({
            'items': items,
            'total': total_count,
            'page': page,
            'pageSize': page_size,
            'pages': (total_count + page_size - 1) // page_size
        })
        
    except Exception as e:
        logger.error(f"获取olam数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/olam/export', methods=['GET'])
def export_olam_data():
    """获取所有olam表数据（用于导出）"""
    try:
        # 获取查询参数
        batch_ids = request.args.getlist('batchIds')  # 获取批次ID参数
        device_ids = request.args.getlist('deviceIds')
        operator_ids = request.args.getlist('operatorIds')
        lot_numbers = request.args.getlist('lotNumbers')
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')
        target_f_min = request.args.get('targetFMin', type=float)
        target_f_max = request.args.get('targetFMax', type=float)
        
        # 构建WHERE条件
        conditions = []
        params = []
        
        # 批次ID条件 - 优先级最高
        if batch_ids:
            batch_placeholders = ', '.join(['%s'] * len(batch_ids))
            conditions.append(f"o.batch_id IN ({batch_placeholders})")
            params.extend(batch_ids)
        # 只有在没有指定批次ID时，才应用其他筛选条件
        else:
            # 多个设备ID条件
            if device_ids:
                device_placeholders = ', '.join(['%s'] * len(device_ids))
                conditions.append(f"o.device_id IN ({device_placeholders})")
                params.extend(device_ids)
            
            # 多个操作员ID条件
            if operator_ids:
                operator_placeholders = ', '.join(['%s'] * len(operator_ids))
                conditions.append(f"o.employee_id IN ({operator_placeholders})")
                params.extend(operator_ids)
                
            # 处理Lot号筛选
            if lot_numbers:
                lot_placeholders = ', '.join(['%s'] * len(lot_numbers))
                conditions.append(f"o.lot IN ({lot_placeholders})")
                params.extend(lot_numbers)
            
            # 日期范围条件
            if start_date:
                conditions.append("DATE(o.start_time) >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("DATE(o.start_time) <= %s")
                params.append(end_date)
            
            # 频率范围条件
            if target_f_min is not None:
                conditions.append("o.target_f >= %s")
                params.append(target_f_min)
            if target_f_max is not None:
                conditions.append("o.target_f <= %s")
                params.append(target_f_max)
        
        # 构建WHERE子句
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # 查询所有符合条件的数据 - 不分页
        data_query = f"""
            SELECT 
                o.batch_id,
                o.lot,
                o.device_id,
                o.employee_id as operator_id,
                o.start_time,
                o.end_time,
                o.start_f,
                o.target_f,
                o.final_f
            FROM olam o
            WHERE {where_clause}
            ORDER BY o.start_time DESC
        """
        
        items = execute_query(data_query, params)
        
        # 如果有数据，再单独查询标准差
        if items:
            # 获取所有批次ID
            batch_ids = [item['batch_id'] for item in items]
            
            # 避免SQL注入，使用参数化查询
            batch_placeholders = ', '.join(['%s'] * len(batch_ids))
            
            # 从last_round_f表中计算标准差
            stdev_query = f"""
                SELECT 
                    batch_id,
                    STDDEV(fre) as stdev
                FROM last_round_f
                WHERE batch_id IN ({batch_placeholders})
                GROUP BY batch_id
            """
            
            stdev_data = {}
            try:
                stdev_results = execute_query(stdev_query, batch_ids)
                if stdev_results:
                    # 创建批次ID到标准差的映射
                    for row in stdev_results:
                        stdev_data[row['batch_id']] = row['stdev']
                    
                    # 更新批次数据
                    for item in items:
                        batch_id = item['batch_id']
                        item['stdev'] = float(stdev_data.get(batch_id, 0.0))
                        
            except Exception as e:
                logger.warning(f"获取标准差数据失败: {str(e)}")
                # 出错时也要继续处理，为所有项目设置默认值
                for item in items:
                    item['stdev'] = 0.0
        
        # 返回结果
        return jsonify(items)
        
    except Exception as e:
        logger.error(f"导出olam数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/last_round_f/export', methods=['GET'])
def export_last_round_f_data():
    """获取last_round_f表数据（用于导出）"""
    try:
        # 获取查询参数，主要是批次ID列表
        batch_ids = request.args.getlist('batchIds')
        
        if not batch_ids:
            return jsonify([])
            
        # 避免SQL注入，使用参数化查询
        batch_placeholders = ', '.join(['%s'] * len(batch_ids))
        
        # 查询数据
        query = f"""
            SELECT 
                batch_id,
                fre
            FROM last_round_f
            WHERE batch_id IN ({batch_placeholders})
            ORDER BY batch_id
        """
        
        results = execute_query(query, batch_ids)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"导出last_round_f数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("正在启动服务器...")
    print("Flask 版本:", flask.__version__)
    print("服务器将运行在: http://localhost:5000")
    app.run(host='127.0.0.1', port=5000, debug=True) 
