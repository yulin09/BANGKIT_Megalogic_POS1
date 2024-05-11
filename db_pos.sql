-- Skema tabel
CREATE TABLE customers (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    customer_name VARCHAR(100),
    gender ENUM('Male', 'Female', 'Other'),
    age INT,
    job VARCHAR(100),
    segment ENUM('Consumer', 'Corporate', 'Home Office'),
    total_spend DECIMAL(10, 2)
);

CREATE TABLE products (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(100),
    product_category VARCHAR(50),
    product_sub_category VARCHAR(50)
);

CREATE TABLE orders (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    order_date DATE,
    ship_date DATE,
    customer_id INT,
    product_id INT, -- Menyimpan ID produk
    FOREIGN KEY (customer_id) REFERENCES customers(ID),
    FOREIGN KEY (product_id) REFERENCES products(ID)
);

INSERT INTO customers (customer_name, gender, age, job, segment, total_spend) VALUES
('Andi Setiawan', 'Male', 29, 'Software Developer', 'Consumer', 15000000.00),
('Budi Raharjo', 'Male', 34, 'Marketing Manager', 'Corporate', 20000000.00),
('Siti Aminah', 'Female', 27, 'HR Specialist', 'Home Office', 7500000.00),
('Rina Kumala', 'Female', 22, 'Student', 'Consumer', 5000000.00),
('Agus Purnomo', 'Male', 45, 'Entrepreneur', 'Corporate', 30000000.00),
('Dewi Sartika', 'Female', 31, 'Data Analyst', 'Home Office', 18000000.00),
('Yusuf Wibisono', 'Male', 36, 'Civil Engineer', 'Consumer', 25000000.00),
('Lia Wulandari', 'Female', 25, 'Graphic Designer', 'Consumer', 10000000.00),
('Fajar Nugraha', 'Male', 40, 'Architect', 'Corporate', 35000000.00),
('Mira Susanti', 'Female', 28, 'Photographer', 'Home Office', 12000000.00);

INSERT INTO products (product_name, product_category, product_sub_category) VALUES
('Lenovo IdeaPad 3', 'Electronics', 'Laptop'),
('Samsung Galaxy S21', 'Electronics', 'Smartphone'),
('Canon EOS 1500D', 'Electronics', 'Camera'),
('Buku Tulis Sinar Dunia', 'Stationery', 'Notebook'),
('Meja Kantor Ergonomis', 'Furniture', 'Office Desk'),
('Kursi Kantor Hitam', 'Furniture', 'Office Chair'),
('Lampu Meja LED Philips', 'Electronics', 'Lighting'),
('Tas Laptop Eiger', 'Fashion', 'Accessories'),
('Hoodie Polos Navy Blue', 'Fashion', 'Clothing'),
('Sepatu Sneakers Adidas', 'Fashion', 'Footwear');

INSERT INTO orders (order_date, ship_date, customer_id, product_id) VALUES
('2023-05-01', '2023-05-03', 1, 1),
('2023-05-02', '2023-05-04', 2, 2),
('2023-05-02', '2023-05-05', 3, 3),
('2023-05-03', '2023-05-06', 4, 4),
('2023-05-04', '2023-05-08', 5, 5),
('2023-05-05', '2023-05-07', 6, 6),
('2023-05-06', '2023-05-08', 7, 7),
('2023-05-07', '2023-05-09', 8, 8),
('2023-05-08', '2023-05-10', 9, 9),
('2023-05-09', '2023-05-11', 10, 10);

INSERT INTO customers (customer_name, gender, age, job, segment, total_spend)
SELECT
    CONCAT('Customer', LPAD(FLOOR(RAND() * 1000), 3, '0')) AS customer_name,
    IF(RAND() < 0.5, 'Male', 'Female') AS gender,
    FLOOR(RAND() * (70 - 18 + 1)) + 18 AS age,
    CASE FLOOR(RAND() * 5)
        WHEN 0 THEN 'Software Developer'
        WHEN 1 THEN 'Marketing Manager'
        WHEN 2 THEN 'HR Specialist'
        WHEN 3 THEN 'Student'
        WHEN 4 THEN 'Entrepreneur'
        WHEN 5 THEN 'Data Analyst'
        WHEN 6 THEN 'Civil Engineer'
        WHEN 7 THEN 'Graphic Designer'
        WHEN 8 THEN 'Architect'
        WHEN 9 THEN 'Photographer'
    END AS job,
    CASE FLOOR(RAND() * 3)
        WHEN 0 THEN 'Consumer'
        WHEN 1 THEN 'Home Office'
        WHEN 2 THEN 'Corporate'
    END AS segment,
    ROUND(RAND() * 1000000, 2) AS total_spend
FROM
    information_schema.tables t1,
    information_schema.tables t2
LIMIT 62;

INSERT INTO products (product_name, product_category, product_sub_category)
SELECT
    CONCAT('Product', LPAD(FLOOR(RAND() * 1000), 3, '0')) AS product_name,
    CASE FLOOR(RAND() * 3)
        WHEN 0 THEN 'Electronics'
        WHEN 1 THEN 'Stationery'
        WHEN 2 THEN 'Furniture'
        WHEN 3 THEN 'Fashion'
    END AS product_category,
    CASE
        WHEN
            (CASE FLOOR(RAND() * 3)
                WHEN 0 THEN 'Electronics'
                WHEN 1 THEN 'Stationery'
                WHEN 2 THEN 'Furniture'
                WHEN 3 THEN 'Fashion'
            END) = 'Electronics' THEN 
            CASE FLOOR(RAND() * 3)
                WHEN 0 THEN 'Laptop'
                WHEN 1 THEN 'Smartphone'
                WHEN 2 THEN 'Camera'
                WHEN 3 THEN 'Lighting'
            END
        WHEN
            (CASE FLOOR(RAND() * 3)
                WHEN 0 THEN 'Electronics'
                WHEN 1 THEN 'Stationery'
                WHEN 2 THEN 'Furniture'
                WHEN 3 THEN 'Fashion'
            END) = 'Stationery' THEN 
            CASE FLOOR(RAND() * 1)
                WHEN 0 THEN 'Notebook'
            END
        WHEN
            (CASE FLOOR(RAND() * 3)
                WHEN 0 THEN 'Electronics'
                WHEN 1 THEN 'Stationery'
                WHEN 2 THEN 'Furniture'
                WHEN 3 THEN 'Fashion'
            END) = 'Furniture' THEN 
            CASE FLOOR(RAND() * 1)
                WHEN 0 THEN 'Office Desk'
                WHEN 1 THEN 'Office Chair'
            END
        WHEN
            (CASE FLOOR(RAND() * 3)
                WHEN 0 THEN 'Electronics'
                WHEN 1 THEN 'Stationery'
                WHEN 2 THEN 'Furniture'
                WHEN 3 THEN 'Fashion'
            END) = 'Fashion' THEN 
            CASE FLOOR(RAND() * 2)
                WHEN 0 THEN 'Accessories'
                WHEN 1 THEN 'Clothing'
                WHEN 2 THEN 'Footwear'
            END
    END AS product_sub_category
FROM
    information_schema.tables t1,
    information_schema.tables t2
LIMIT 700;




INSERT INTO orders (order_date, ship_date, customer_id, product_id)
SELECT
    DATE_ADD('2023-05-01', INTERVAL FLOOR(RAND() * 365) DAY) AS order_date,
    DATE_ADD('2023-05-01', INTERVAL FLOOR(RAND() * 10) DAY) AS ship_date,
    FLOOR(RAND() * 100) + 1 AS customer_id,
    FLOOR(RAND() * 10) + 1 AS product_id
FROM
    information_schema.tables t1,
    information_schema.tables t2
LIMIT 688;

