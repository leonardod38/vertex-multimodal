import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def somar(a: float, b: float) -> float:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Parâmetros devem ser numéricos")
    resultado = a + b
    logger.info(f"somar({a}, {b}) = {resultado}")
    return resultado


if __name__ == "__main__":
    logger.info("Iniciando exemplo")
    print(somar(3, 7))
    print(somar(10.5, 4.5))
    logger.info("Concluído")
