import psycopg2
from dotenv import load_dotenv
import os
import logging

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

def init_db():
    """初始化数据库表"""
    try:
        # 连接数据库
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # 创建设备表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id SERIAL PRIMARY KEY,
                device_name VARCHAR(100) NOT NULL
            );
        """)
        logger.info("设备表创建成功")
        
        # 创建操作员表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS operators (
                operator_id SERIAL PRIMARY KEY,
                operator_name VARCHAR(100) NOT NULL
            );
        """)
        logger.info("操作员表创建成功")
        
        # 创建批次表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                batch_id SERIAL PRIMARY KEY,
                device_id INTEGER REFERENCES devices(device_id),
                operator_id INTEGER REFERENCES operators(operator_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resonant FLOAT,
                stdev FLOAT
            );
        """)
        logger.info("批次表创建成功")
        
        # 提交事务
        conn.commit()
        logger.info("所有表创建成功")
        
        # 插入一些测试数据
        # 插入设备
        cur.execute("""
            INSERT INTO devices (device_name) 
            VALUES ('设备1'), ('设备2'), ('设备3')
            ON CONFLICT DO NOTHING;
        """)
        
        # 插入操作员
        cur.execute("""
            INSERT INTO operators (operator_name) 
            VALUES ('操作员1'), ('操作员2'), ('操作员3')
            ON CONFLICT DO NOTHING;
        """)
        
        # 插入一些批次数据
        cur.execute("""
            INSERT INTO batches (device_id, operator_id, resonant, stdev)
            VALUES 
                (1, 1, 100.5, 0.5),
                (2, 2, 101.2, 0.3),
                (3, 3, 99.8, 0.4),
                (1, 2, 100.2, 0.6),
                (2, 3, 101.0, 0.2),
                (3, 1, 99.5, 0.3),
                (1, 3, 100.8, 0.4),
                (2, 1, 101.5, 0.7),
                (3, 2, 99.9, 0.5),
                (1, 1, 100.3, 0.4)
            ON CONFLICT DO NOTHING;
        """)
        
        # 提交事务
        conn.commit()
        logger.info("测试数据插入成功")
        
    except Exception as e:
        logger.error(f"初始化数据库失败: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    init_db() 