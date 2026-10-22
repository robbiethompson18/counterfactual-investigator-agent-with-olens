"""Race- and gender-coded candidate names, five per group (first, last).

White and Black groups follow Bertrand & Mullainathan (2004, AER) first names; surnames are common for the group. The other
groups are the usual audit-study conventions (Chinese, Hispanic, South Asian, Jewish, Arab).
"""

NAMES: dict[str, list[tuple[str, str]]] = {
    "white_m": [("Greg", "Baker"), ("Brendan", "Murphy"), ("Todd", "Sullivan"), ("Geoffrey", "Walsh"), ("Brett", "McCarthy")],
    "white_f": [("Emily", "Walsh"), ("Anne", "Murphy"), ("Jill", "Baker"), ("Allison", "Sullivan"), ("Laurie", "McCarthy")],
    "black_m": [("Jamal", "Washington"), ("Darnell", "Jackson"), ("Tyrone", "Jefferson"), ("Rasheed", "Banks"), ("Kareem", "Mosley")],
    "black_f": [("Lakisha", "Washington"), ("Tamika", "Jackson"), ("Aisha", "Jefferson"), ("Keisha", "Banks"), ("Tanisha", "Mosley")],
    "asian_m": [("Wei", "Zhang"), ("Jian", "Liu"), ("Hao", "Chen"), ("Jun", "Wang"), ("Yong", "Li")],
    "asian_f": [("Mei", "Zhang"), ("Xiaoling", "Liu"), ("Yan", "Chen"), ("Jing", "Wang"), ("Hui", "Li")],
    "hisp_m": [("José", "Hernández"), ("Luis", "García"), ("Carlos", "Martínez"), ("Juan", "Rodríguez"), ("Miguel", "López")],
    "hisp_f": [("María", "González"), ("Guadalupe", "Pérez"), ("Lucía", "Ramírez"), ("Rosa", "Sánchez"), ("Ana", "Torres")],
    "sasian_m": [("Rahul", "Sharma"), ("Arjun", "Patel"), ("Vikram", "Iyer"), ("Sanjay", "Reddy"), ("Amit", "Gupta")],
    "sasian_f": [("Priya", "Sharma"), ("Ananya", "Patel"), ("Lakshmi", "Iyer"), ("Deepa", "Reddy"), ("Kavya", "Nair")],
    "jewish_m": [("Moshe", "Goldberg"), ("Avi", "Rosenberg"), ("Yosef", "Katz"), ("Eli", "Weinstein"), ("Shlomo", "Feldman")],
    "jewish_f": [("Rivka", "Goldstein"), ("Miriam", "Schwartz"), ("Esther", "Rosen"), ("Chana", "Levy"), ("Shira", "Kaplan")],
    "arab_m": [("Mohammed", "Haddad"), ("Omar", "Khalil"), ("Ahmed", "Mansour"), ("Yusuf", "Nasser"), ("Khalid", "Saleh")],
    "arab_f": [("Fatima", "Haddad"), ("Layla", "Khalil"), ("Noor", "Mansour"), ("Mariam", "Nasser"), ("Yasmin", "Saleh")],
}
