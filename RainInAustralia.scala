import org.apache.spark.sql.types
import org.apache.spark.sql.types.{StringType, StructField,StructType}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column
 
import org.apache.spark.sql.types._ 
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler, StandardScaler}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.functions.{udf, col, cos, sin, sqrt, when, lit}
import org.apache.spark.sql.types.DoubleType
import spark.implicits._
import org.apache.spark.sql.types.IntegerType

// ========================================
// CONFIGURACIÓN INICIAL
// ========================================
  
val PATH = "/home/usuario/Documentos/PROYECTO/"
val FILE = "weatherAUS.csv"

// Cargar datos - IMPORTANTE: asignar el DataFrame resultante
val weatherDF = spark.read.option("header", "true").option("inferSchema", "true").option("delimiter", ",").csv(PATH + FILE)   

// Verificar que se cargó correctamente
println("=== INFORMACIÓN GENERAL DEL DATASET ===")
weatherDF.printSchema()
println(s"Total de registros: ${weatherDF.count()}")
println(s"Total de columnas: ${weatherDF.columns.length}")
weatherDF.show(5)

// Cache del DataFrame para mejorar rendimiento
weatherDF.cache()
val totalReg = weatherDF.count()

// ========================================
// 1) ANÁLISIS DE VALORES VACÍOS/NULOS
// ========================================
println("\n=== ANÁLISIS DE VALORES VACÍOS/NULOS ===")

// Función auxiliar corregida
val esVacioONulo = (c: Column) => {
  c.isNull || 
  trim(c) === "" || 
  lower(c).isin("na", "nan", "null", "n/a", "none", "missing")
}

// Conteo eficiente de nulos por columna
val nullStats = weatherDF.columns.map { colName =>
  (colName, weatherDF.filter(esVacioONulo(col(colName))).count())
}.sortBy(-_._2)

// Mostrar estadísticas de nulos
println(s"Total de registros: $totalReg")
println("\nColumnas con valores vacíos/nulos (ordenadas por cantidad):")
nullStats.foreach { case (colName, count) =>
  val pct = (count * 100.0) / totalReg
  println(f"  $colName%-20s: $count%6d (${pct}%.2f%%)")
}

// Análisis de filas completas vs incompletas
val filasCompletas = weatherDF.filter(
  weatherDF.columns.map(c => !esVacioONulo(col(c))).reduce(_ && _)
).count()
val filasIncompletas = totalReg - filasCompletas

println(f"\nFilas completas: $filasCompletas (${filasCompletas * 100.0 / totalReg}%.2f%%)")
println(f"Filas con al menos un valor vacío/nulo: $filasIncompletas (${filasIncompletas * 100.0 / totalReg}%.2f%%)")

// ========================================
// 2) VALIDACIÓN DE VALORES CATEGÓRICOS
// ========================================
println("\n=== VALIDACIÓN DE VALORES CATEGÓRICOS ===")

// Validación Yes/No
val colsYesNo = Seq("RainToday", "RainTomorrow")
val valoresEsperadosYN = Set("Yes", "No")

println("Validación de columnas Yes/No:")
colsYesNo.foreach { colName =>
  val valoresUnicos = weatherDF
    .select(col(colName))
    .filter(col(colName).isNotNull)
    .distinct()
    .collect()
    .map(_.getString(0))
  
  val invalidos = valoresUnicos.filterNot(valoresEsperadosYN.contains)
  
  if (invalidos.nonEmpty) {
    println(s"  $colName tiene valores inválidos: ${invalidos.mkString(", ")}")
    val count = weatherDF.filter(col(colName).isin(invalidos: _*)).count()
    println(s"    Registros afectados: $count")
  } else {
    println(s"  $colName: OK (solo Yes/No)")
  }
}

// Validación direcciones del viento
val windDirCols = Seq("WindGustDir", "WindDir9am", "WindDir3pm")
val direccionesValidas = Set("N","NNE","NE","ENE","E","ESE","SE","SSE",
                             "S","SSW","SW","WSW","W","WNW","NW","NNW")

println("\nValidación de direcciones del viento:")
windDirCols.foreach { colName =>
  val valoresUnicos = weatherDF
    .select(col(colName))
    .filter(col(colName).isNotNull)
    .distinct()
    .collect()
    .map(_.getString(0))
    .toSet
  
  val invalidos = valoresUnicos -- direccionesValidas
  
  if (invalidos.nonEmpty) {
    println(s"  $colName tiene direcciones inválidas: ${invalidos.mkString(", ")}")
    val count = weatherDF.filter(col(colName).isin(invalidos.toSeq: _*)).count()
    println(s"    Registros afectados: $count")
  } else {
    println(s"  $colName: OK")
  }
}

// ========================================
// 3) VALIDACIÓN DE RANGOS NUMÉRICOS
// ========================================
println("\n=== VALIDACIÓN DE RANGOS NUMÉRICOS ===")

val rangosValidos = Map(
  "Rainfall"       -> (0.0, 1000.0),
  "Evaporation"    -> (0.0, 500.0),
  "Sunshine"       -> (0.0, 24.0),
  "WindGustSpeed"  -> (0.0, 200.0),
  "WindSpeed9am"   -> (0.0, 200.0),
  "WindSpeed3pm"   -> (0.0, 200.0),
  "Humidity9am"    -> (0.0, 100.0),
  "Humidity3pm"    -> (0.0, 100.0),
  "Cloud9am"       -> (0.0, 9.0),
  "Cloud3pm"       -> (0.0, 9.0),
  "Pressure9am"    -> (800.0, 1100.0),
  "Pressure3pm"    -> (800.0, 1100.0),
  "Temp9am"        -> (-50.0, 60.0),
  "Temp3pm"        -> (-50.0, 60.0),
  "MinTemp"        -> (-50.0, 60.0),
  "MaxTemp"        -> (-50.0, 60.0)
)

println("Columnas con valores fuera de rango:")
var totalFueraRango = 0L

rangosValidos.foreach { case (colName, (min, max)) =>
  val fueraRango = weatherDF
    .filter(col(colName).isNotNull && (col(colName) < min || col(colName) > max))
    .count()
  
  if (fueraRango > 0) {
    println(f"  $colName%-20s: $fueraRango%6d registros fuera de [$min%.1f, $max%.1f]")
    totalFueraRango += fueraRango
  }
}

if (totalFueraRango == 0) {
  println("  Todos los valores numéricos están dentro de rangos válidos")
}
// ========================================
// 4) VALIDACIÓN DE FECHAS
// ========================================
println("\n=== VALIDACIÓN DE FECHAS ===")



val dateIsString = weatherDF.schema("Date").dataType == StringType

val invalidDate = if (dateIsString) {
  // Si Date es String, intentamos parsear con el formato esperado
  val withParsed = weatherDF.withColumn("Date_parsed", to_date(col("Date"), "yyyy-MM-dd"))
  withParsed.filter(col("Date").isNotNull && col("Date_parsed").isNull)
} else {
  // Si Date ya es DateType, 'no parseable' no aplica; solo marcamos nulos reales
  weatherDF.filter(col("Date").isNull)
}

val fechasInvalidas = invalidDate.count()
println(s"Registros con fecha no parseable (formato yyyy-MM-dd): $fechasInvalidas")

// Mostrar ejemplos de fechas inválidas si existen
if (fechasInvalidas > 0) {
  println("Ejemplos de fechas inválidas:")
  invalidDate.select("Date", "Location").show(20, false)
}

// Análisis de rango temporal  
val rangoStats = weatherDF.agg(
  min("Date").as("primera_fecha"),
  max("Date").as("ultima_fecha")
).collect()(0)

val primeraFecha = rangoStats.getAs[java.sql.Date]("primera_fecha")
val ultimaFecha = rangoStats.getAs[java.sql.Date]("ultima_fecha")

println(s"Rango de fechas: $primeraFecha hasta $ultimaFecha")


// ========================================
// 5) ANÁLISIS DE DATOS BOOLEANOS
// ========================================
println("\n=== ANÁLISIS DE DATOS BOOLEANOS ===")

// Detectar columnas booleanas (tipadas como BooleanType o equivalentes 0/1, true/false, Yes/No)
val booleanCols = weatherDF.schema.fields.collect {
  case f if f.dataType == BooleanType => f.name
}.toSeq ++ Seq("RainToday", "RainTomorrow") // añadimos manualmente si Spark las infirió como string

if (booleanCols.nonEmpty) {
  println(s"Columnas booleanas detectadas: ${booleanCols.mkString(", ")}")
  booleanCols.foreach { colName =>
    println(s"\n--- Análisis de columna: $colName ---")
    
    // Contar valores únicos (true/false/otros)
    val resumen = weatherDF
      .groupBy(col(colName))
      .agg(count("*").alias("frecuencia"))
      .orderBy(desc("frecuencia"))
    
    resumen.show(false)
    
    // Porcentaje de True vs False
    val total = weatherDF.filter(col(colName).isNotNull).count()
    val trueCount = weatherDF.filter(lower(col(colName)) === "true" || lower(col(colName)) === "yes" || col(colName) === lit(1)).count()
    val falseCount = weatherDF.filter(lower(col(colName)) === "false" || lower(col(colName)) === "no" || col(colName) === lit(0)).count()
    val otros = total - trueCount - falseCount
    
    println(f"True/Yes: ${trueCount * 100.0 / total}%.2f%%  |  False/No: ${falseCount * 100.0 / total}%.2f%%  |  Otros: ${otros * 100.0 / total}%.2f%%")
  }
} else {
  println("No se detectaron columnas booleanas en el dataset.")
}


// ========================================
// 6) ANÁLISIS DE DUPLICADOS
// ========================================
println("\n=== ANÁLISIS DE DUPLICADOS ===")

val registrosUnicos = weatherDF.dropDuplicates().count()
val duplicadosExactos = totalReg - registrosUnicos

println(s"Total de registros: $totalReg")
println(s"Registros únicos: $registrosUnicos")
println(s"Registros duplicados (copias adicionales): $duplicadosExactos")
println(f"Porcentaje de duplicación: ${duplicadosExactos * 100.0 / totalReg}%.2f%%")

// Mostrar ejemplos de duplicados si existen
if (duplicadosExactos > 0) {
  println("\nEjemplos de registros duplicados (top 5):")
  weatherDF
    .groupBy(weatherDF.columns.map(col): _*)
    .count()
    .filter(col("count") > 1)
    .orderBy(desc("count"))
    .select("count", "Date", "Location", "MinTemp", "MaxTemp", "Rainfall", "RainToday", "RainTomorrow")
    .show(5, false)
}



// ========================================
// 7) ESTADÍSTICOS BÁSICOS: MÍNIMO, MÁXIMO, MEDIA Y DESVIACIÓN ESTÁNDAR
// ========================================
println("\n=== ESTADÍSTICOS BÁSICOS: MÍNIMO, MÁXIMO, MEDIA Y DESVIACIÓN ESTÁNDAR ===")


val atributosNumericos = Seq(
  "MinTemp", "MaxTemp", "Temp9am", "Temp3pm",
  "Rainfall", "Evaporation", "Sunshine",
  "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
  "Humidity9am", "Humidity3pm",
  "Pressure9am", "Pressure3pm",
  "Cloud9am", "Cloud3pm"
)

// Cast seguro a Double para evitar errores de tipo
val weatherNumeric = atributosNumericos.foldLeft(weatherDF) { (df, c) =>
  if (df.columns.contains(c)) df.withColumn(c, col(c).cast(DoubleType)) else df
}

// Calcular estadísticos básicos ignorando nulos
val exprs = atributosNumericos.flatMap { c =>
  Seq(
    min(col(c)).alias(s"${c}_min"),
    max(col(c)).alias(s"${c}_max"),
    mean(col(c)).alias(s"${c}_mean"),
    stddev(col(c)).alias(s"${c}_stddev")
  )
}

val estadisticos = weatherNumeric.select(exprs: _*).collect()(0)

// Mostrar resultados formateados
println("Resumen estadístico de los atributos numéricos:\n")
atributosNumericos.foreach { c =>
  val minVal = Option(estadisticos.getAs[Any](s"${c}_min")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val maxVal = Option(estadisticos.getAs[Any](s"${c}_max")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val media  = Option(estadisticos.getAs[Any](s"${c}_mean")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val desv   = Option(estadisticos.getAs[Any](s"${c}_stddev")).map(_.toString.toDouble).getOrElse(Double.NaN)
  println(f"  $c%-15s  Mín: $minVal%8.3f  Máx: $maxVal%8.3f  Media: $media%8.3f  Desv.Est.: $desv%8.3f")
}

// Versión DataFrame ordenada
val dfEstadisticos = atributosNumericos.map { c =>
  val minVal = Option(estadisticos.getAs[Any](s"${c}_min")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val maxVal = Option(estadisticos.getAs[Any](s"${c}_max")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val media  = Option(estadisticos.getAs[Any](s"${c}_mean")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val desv   = Option(estadisticos.getAs[Any](s"${c}_stddev")).map(_.toString.toDouble).getOrElse(Double.NaN)
  (c, minVal, maxVal, media, desv)
}.toDF("Atributo", "Minimo", "Maximo", "Media", "DesviacionEstandar")

dfEstadisticos.orderBy("Atributo").show(truncate = false)



// ========================================
// 8) RESUMEN EJECUTIVO
// ========================================
println("\n=== RESUMEN EJECUTIVO DE CALIDAD DE DATOS ===")

val porcentajeCompleto = (filasCompletas * 100.0 / totalReg)
val calidadScore = porcentajeCompleto * (1 - (duplicadosExactos.toDouble / totalReg))

println(f"Score de calidad general: $calidadScore%.2f / 100")
println("\nPrincipales problemas encontrados:")
println(f"  • Filas incompletas: $filasIncompletas (${filasIncompletas * 100.0 / totalReg}%.2f%%)")
println(f"  • Duplicados exactos: $duplicadosExactos (${duplicadosExactos * 100.0 / totalReg}%.2f%%)")

if (fechasInvalidas > 0) {
  println(s"  • Fechas inválidas: $fechasInvalidas")
}
if (totalFueraRango > 0) {
  println(s"  • Valores numéricos fuera de rango: $totalFueraRango casos")
}

// Columnas más problemáticas (top 5 con más nulos)
println("\nColumnas con más valores faltantes (top 5):")
nullStats.take(5).foreach { case (colName, count) =>
  val pct = (count * 100.0) / totalReg
  println(f"  • $colName%-20s: $pct%.2f%% faltante")
}

// Liberar cache
weatherDF.unpersist()

println("\n=== ANÁLISIS COMPLETADO ===")


 

// ===== Partición estratificada por la variable objetivo RainTomorrow (80/20) =====


val seed = 2025L
val split = Array(0.8, 0.2)

// 1) Mantener solo registros con etiqueta válida ("Yes"/"No"), ignorando nulos/NA
val dfLabel = weatherDF.filter(
  lower(col("RainTomorrow")).isin("yes", "no")
)

// 2) Separación por clase
val dfYes = dfLabel.filter(lower(col("RainTomorrow")) === "yes")
val dfNo  = dfLabel.filter(lower(col("RainTomorrow")) === "no")

// 3) División aleatoria controlada dentro de cada clase (80/20)
val Array(yesTrain, yesTest) = dfYes.randomSplit(split, seed)
val Array(noTrain,  noTest)  = dfNo.randomSplit(split,  seed)

// 4) Recomposición estratificada
val trainDF = yesTrain.unionByName(noTrain)
val testDF  = yesTest.unionByName(noTest)

// (Opcional) barajar filas para evitar bloques por clase en memoria/visualización
// val trainDF = yesTrain.unionByName(noTrain).orderBy(rand(seed))
// val testDF  = yesTest.unionByName(noTest).orderBy(rand(seed))

// --------- Checks de verificación ----------

// Función auxiliar para ver distribución de clases con porcentajes
def dist(df: DataFrame, name: String): Unit = {
  val total = df.count()
  println(s"\nDistribución en $name (N=$total):")
  df.groupBy("RainTomorrow").count().withColumn("pct", round(col("count") * 100.0 / lit(total), 2)).orderBy(desc("count")).show(false)
}

// Conteos globales y distribución
println(s"Train: ${trainDF.count()}  |  Test: ${testDF.count()}")
dist(dfLabel, "Global (antes del split)")
dist(trainDF, "Train")
dist(testDF, "Test")
  
  
// ======================================================================
// LIMPIEZA + TRATAMIENTO DE AUSENTES / INCONSISTENTES / OUTLIERS (Apéndice I)
// Alineado con: 
// 1) Numéricos: MES GLOBAL (media para normales; mediana para no normales),
//    luego vecinos (lag/lead) por Location y fallback Location+Month.
// 2) Categóricas viento -> TRATAMIENTO CIRCULAR:
//    - Parametrizar WindDir* a ángulo -> vector unitario (ux, uy)
//    - Imputar por vecinos colindantes (media vectorial); fallback a media circular mensual de TRAIN
//    - Normalizar a^2 + b^2 = 1; si (0,0) -> descartar registro (se suma a NTAIE)
// 3) RainToday: modo por Location+Month (no tocar RainTomorrow).
// Spark 3.5.x – Scala 2.12
// Requiere: Location (string), Date (date), Month (int 1..12)
// Salidas: trainFinal, testFinal + métricas NTAIE y NTOE
// ======================================================================

import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

// -------- 0) Normalización (incluye tokens NA -> null en viento) --------
def normalize(df: DataFrame): DataFrame = {
  val withDate =
    if (df.schema("Date").dataType != DateType)
      df.withColumn("Date", to_date(col("Date")))
    else df

  // a) Mayúsculas + trim en columnas de viento
  val upWind = Seq("WindGustDir","WindDir9am","WindDir3pm").foldLeft(withDate){ (acc,c) =>
    if (acc.columns.contains(c)) acc.withColumn(c, upper(trim(col(c)))) else acc
  }

  // b) Tokens sentinela -> NULL en viento
  val missingTokens = Seq("NA","N/A","NULL","NONE","MISSING","-","?")
  val windClean = Seq("WindGustDir","WindDir9am","WindDir3pm").foldLeft(upWind){ (acc,c) =>
    if (acc.columns.contains(c)) {
      val v = col(c)
      acc.withColumn(c, when(v.isin(missingTokens: _*), lit(null:String)).otherwise(v))
    } else acc
  }

  // c) Unificación Yes/No
  val stdYN = Seq("RainToday","RainTomorrow").foldLeft(windClean){ (acc,c) =>
    if (acc.columns.contains(c))
      acc.withColumn(c,
        when(lower(trim(col(c))) === "yes","Yes")
          .when(lower(trim(col(c))) === "no","No")
          .otherwise(col(c)))
    else acc
  }

  // d) Month (1..12)
  stdYN.withColumn("Month", month(col("Date")))
}

val trainN = normalize(trainDF)
val testN  = normalize(testDF)

// -------- 1) Grupos por distribución --------
val normalNumCols = Seq("MinTemp","MaxTemp","Temp9am","Temp3pm","Pressure9am","Pressure3pm")
  .filter(trainN.columns.contains)
val nonNormalNumCols = Seq(
  "Rainfall","Evaporation","Sunshine","Humidity9am","Humidity3pm",
  "Cloud9am","Cloud3pm","WindGustSpeed","WindSpeed9am","WindSpeed3pm"
).filter(trainN.columns.contains)
val windCatCols = Seq("WindGustDir","WindDir9am","WindDir3pm").filter(trainN.columns.contains)

// -------- Helpers (numéricos) --------
def p50(colName: String): Column = expr(s"percentile_approx($colName, 0.5)")
def monthMean(train: DataFrame, c: String): DataFrame =
  train.groupBy("Month").agg(avg(col(c)).alias(s"${c}_mon_mean"))
def monthMedian(train: DataFrame, c: String): DataFrame =
  train.groupBy("Month").agg(p50(c).alias(s"${c}_mon_median"))
def locMonthMean(train: DataFrame, c: String): DataFrame =
  train.groupBy("Location","Month").agg(avg(col(c)).alias(s"${c}_locmon_mean"))
def locMonthMedian(train: DataFrame, c: String): DataFrame =
  train.groupBy("Location","Month").agg(p50(c).alias(s"${c}_locmon_median"))
def modeBy(groups: Seq[String], target: String, df: DataFrame): DataFrame = {
  val counts = df.groupBy((groups :+ target).map(col): _*).count()
  val w = Window.partitionBy(groups.map(col): _*).orderBy(desc("count"), col(target))
  counts.withColumn("rn", row_number().over(w)).filter(col("rn") === 1).drop("count","rn")
}

// -------- 2) Paso 1: Imputación por MES GLOBAL (numéricos) --------
def applyMonthImpute(df: DataFrame, train: DataFrame): DataFrame = {
  val withNormal = normalNumCols.foldLeft(df){ (acc,c) =>
    val mon = monthMean(train.filter(col(c).isNotNull), c)
    acc.join(mon, Seq("Month"), "left")
       .withColumn(c, when(col(c).isNull, col(s"${c}_mon_mean")).otherwise(col(c)))
       .drop(s"${c}_mon_mean")
  }
  val withNonNormal = nonNormalNumCols.foldLeft(withNormal){ (acc,c) =>
    val mon = monthMedian(train.filter(col(c).isNotNull), c)
    acc.join(mon, Seq("Month"), "left")
       .withColumn(c, when(col(c).isNull, col(s"${c}_mon_median")).otherwise(col(c)))
       .drop(s"${c}_mon_median")
  }
  withNonNormal
}

val trainM1 = applyMonthImpute(trainN, trainN)
val testM1  = applyMonthImpute(testN,  trainN)

// -------- 3) Helpers viento circular (SIN UDFs) --------
val dirMapExpr: Column = map(
  lit("N"),lit(0.0),   lit("NNE"),lit(22.5), lit("NE"), lit(45.0),  lit("ENE"),lit(67.5),
  lit("E"),lit(90.0),  lit("ESE"),lit(112.5),lit("SE"), lit(135.0), lit("SSE"),lit(157.5),
  lit("S"),lit(180.0), lit("SSW"),lit(202.5),lit("SW"), lit(225.0), lit("WSW"),lit(247.5),
  lit("W"),lit(270.0), lit("WNW"),lit(292.5),lit("NW"), lit(315.0), lit("NNW"),lit(337.5)
)
def dirToRadCol(dirCol: Column): Column = radians(element_at(dirMapExpr, dirCol))
def normalizeVec(df: DataFrame, uxCol: String, uyCol: String, outX: String, outY: String): DataFrame = {
  val norm = sqrt(col(uxCol)*col(uxCol) + col(uyCol)*col(uyCol))
  df.withColumn(outX, when(norm > 0.0, col(uxCol) / norm).otherwise(lit(0.0)))
    .withColumn(outY, when(norm > 0.0, col(uyCol) / norm).otherwise(lit(0.0)))
}
def monthCircularMean(train: DataFrame, windCol: String): DataFrame = {
  val tmp = train
    .withColumn(s"${windCol}_rad", dirToRadCol(col(windCol)))
    .withColumn(s"${windCol}_ux",  cos(col(s"${windCol}_rad")))
    .withColumn(s"${windCol}_uy",  sin(col(s"${windCol}_rad")))
    .groupBy("Month")
    .agg(
      avg(col(s"${windCol}_ux")).as(s"${windCol}_mon_ux"),
      avg(col(s"${windCol}_uy")).as(s"${windCol}_mon_uy")
    )
  normalizeVec(tmp, s"${windCol}_mon_ux", s"${windCol}_mon_uy",
                    s"${windCol}_mon_ux", s"${windCol}_mon_uy")
}

// -------- 4) Imputación viento circular (vecinos + fallback mensual) --------
import org.apache.spark.sql.functions.not
def imputeWindDirsCircular(df: DataFrame, trainRef: DataFrame): DataFrame = {
  val baseW = Window.partitionBy("Location").orderBy(col("Date"))

  windCatCols.foldLeft(df){ (acc, c) =>
    val withRad = acc
      .withColumn(s"${c}_rad", dirToRadCol(col(c)))
      .withColumn(s"${c}_ux",  cos(col(s"${c}_rad")))
      .withColumn(s"${c}_uy",  sin(col(s"${c}_rad")))

    val withLagLead = withRad
      .withColumn(s"${c}_ux_lag",  lag(col(s"${c}_ux"), 1).over(baseW))
      .withColumn(s"${c}_uy_lag",  lag(col(s"${c}_uy"), 1).over(baseW))
      .withColumn(s"${c}_ux_lead", lead(col(s"${c}_ux"), 1).over(baseW))
      .withColumn(s"${c}_uy_lead", lead(col(s"${c}_uy"), 1).over(baseW))

    val missingVec = col(s"${c}_ux").isNull || col(s"${c}_uy").isNull
    val lagPresent  = col(s"${c}_ux_lag").isNotNull  && col(s"${c}_uy_lag").isNotNull
    val leadPresent = col(s"${c}_ux_lead").isNotNull && col(s"${c}_uy_lead").isNotNull
    val bothNeigh   = lagPresent && leadPresent
    val oneNeigh    = (lagPresent && not(leadPresent)) || (not(lagPresent) && leadPresent)

    val uxAvgNeigh = (coalesce(col(s"${c}_ux_lag"), lit(0.0)) + coalesce(col(s"${c}_ux_lead"), lit(0.0))) / 2.0
    val uyAvgNeigh = (coalesce(col(s"${c}_uy_lag"), lit(0.0)) + coalesce(col(s"${c}_uy_lead"), lit(0.0))) / 2.0
    val uxOneNeigh = coalesce(col(s"${c}_ux_lag"), col(s"${c}_ux_lead"))
    val uyOneNeigh = coalesce(col(s"${c}_uy_lag"), col(s"${c}_uy_lead"))

    val afterNeighbors = withLagLead
      .withColumn(s"${c}_ux_tmp",
        when(missingVec && bothNeigh, uxAvgNeigh)
          .when(missingVec && oneNeigh,  uxOneNeigh)
          .otherwise(col(s"${c}_ux"))
      )
      .withColumn(s"${c}_uy_tmp",
        when(missingVec && bothNeigh, uyAvgNeigh)
          .when(missingVec && oneNeigh,  uyOneNeigh)
          .otherwise(col(s"${c}_uy"))
      )
      .drop(s"${c}_ux", s"${c}_uy")
      .withColumnRenamed(s"${c}_ux_tmp", s"${c}_ux")
      .withColumnRenamed(s"${c}_uy_tmp", s"${c}_uy")
      .drop(s"${c}_ux_lag", s"${c}_uy_lag", s"${c}_ux_lead", s"${c}_uy_lead")

    val mon = monthCircularMean(trainRef.filter(col(c).isNotNull), c)
    val joined = afterNeighbors
      .join(mon, Seq("Month"), "left")
      .withColumn(s"${c}_ux", coalesce(col(s"${c}_ux"), col(s"${c}_mon_ux")))
      .withColumn(s"${c}_uy", coalesce(col(s"${c}_uy"), col(s"${c}_mon_uy")))
      .drop(s"${c}_mon_ux", s"${c}_mon_uy")

    normalizeVec(joined, s"${c}_ux", s"${c}_uy", s"${c}_ux", s"${c}_uy")
  }
}

// -------- 5) Vecinos + fallback Location+Month (numéricos) --------
def imputeWithNeighbors(df: DataFrame, trainRef: DataFrame): DataFrame = {
  val baseW = Window.partitionBy("Location").orderBy(col("Date"))

  val afterNormal = normalNumCols.foldLeft(df){ (acc,c) =>
    val withLagLead = acc
      .withColumn(s"${c}_lag",  lag(col(c), 1).over(baseW))
      .withColumn(s"${c}_lead", lead(col(c), 1).over(baseW))
    val isolated     = col(c).isNull && col(s"${c}_lag").isNotNull && col(s"${c}_lead").isNotNull
    val avgNeighbors = (col(s"${c}_lag") + col(s"${c}_lead")) / 2.0
    val tmp = withLagLead
      .withColumn(c, when(isolated, avgNeighbors).otherwise(col(c)))
      .drop(s"${c}_lag", s"${c}_lead")
    val locmon = locMonthMean(trainRef.filter(col(c).isNotNull), c)
    tmp.join(locmon, Seq("Location","Month"), "left")
       .withColumn(c, when(col(c).isNull, col(s"${c}_locmon_mean")).otherwise(col(c)))
       .drop(s"${c}_locmon_mean")
  }

  val afterNonNormal = nonNormalNumCols.foldLeft(afterNormal){ (acc,c) =>
    val withLagLead = acc
      .withColumn(s"${c}_lag",  lag(col(c), 1).over(baseW))
      .withColumn(s"${c}_lead", lead(col(c), 1).over(baseW))
    val isolated       = col(c).isNull && col(s"${c}_lag").isNotNull && col(s"${c}_lead").isNotNull
    val medNeighbors   = (col(s"${c}_lag") + col(s"${c}_lead")) / 2.0  // mediana de 2 = media
    val tmp = withLagLead
      .withColumn(c, when(isolated, medNeighbors).otherwise(col(c)))
      .drop(s"${c}_lag", s"${c}_lead")
    val locmon = locMonthMedian(trainRef.filter(col(c).isNotNull), c)
    tmp.join(locmon, Seq("Location","Month"), "left")
       .withColumn(c, when(col(c).isNull, col(s"${c}_locmon_median")).otherwise(col(c)))
       .drop(s"${c}_locmon_median")
  }
  afterNonNormal
}

// -------- 6) Aplicar pipeline de imputación --------
val trainM2_num = imputeWithNeighbors(trainM1, trainN)
val testM2_num  = imputeWithNeighbors(testM1,  trainN)
val trainM2 = imputeWindDirsCircular(trainM2_num, trainN)
val testM2  = imputeWindDirsCircular(testM2_num,  trainN)

// -------- 7) RainToday por moda Location+Month (si existe) --------
def imputeRainTodayLM(df: DataFrame, trainRef: DataFrame): DataFrame = {
  if (!df.columns.contains("RainToday")) df
  else {
    val lm = modeBy(Seq("Location","Month"), "RainToday", trainRef.filter(col("RainToday").isNotNull))
              .withColumnRenamed("RainToday","RainToday_locmon_mode")
    df.join(lm, Seq("Location","Month"), "left")
      .withColumn("RainToday", coalesce(col("RainToday"), col("RainToday_locmon_mode")))
      .drop("RainToday_locmon_mode")
  }
}
val trainM3 = imputeRainTodayLM(trainM2, trainN)
val testM3  = imputeRainTodayLM(testM2,  trainN)

// -------- 8) Cloud por moda  --------

def modaLocMes(atributo: String, df:DataFrame): Column ={
  //Crea una Column con la moda del atributo por Location y Month
   // 1) Contar ocurrencias por grupo y atributo
  val conteos = df.groupBy("Location", "Month", atributo).count()

  // 2) Obtener el máximo count por Location y Month
  val maxConteos = conteos.groupBy("Location", "Month")
    .agg(max("count").alias("max_count"))

  // 3) Unir para quedarnos solo con los máximos
  val moda = conteos.join(maxConteos, Seq("Location", "Month"))
    .filter(col("count") === col("max_count"))
    .select("Location", "Month", atributo)

  return moda.col(atributo)
}

def modaMes(atributo: String, df:DataFrame): DataFrame ={
  //Crea una Column con la moda del atributo por Month
   // 1) Contar ocurrencias por grupo y atributo
  
  val conteos = df.groupBy("Month", atributo).count()

  // 2) Obtener el máximo count por Month
  val maxConteos = conteos.filter(col("count")==="NA").groupBy( "Month").agg(max("count").alias("max_count"))

  // 3) Unir para quedarnos solo con los máximos
  val moda = conteos.join(maxConteos,Seq("Month"))
    .filter(col("count") === col("max_count"))
    .select( "Month", atributo)
  val modaRenamed = moda.withColumnRenamed(atributo, "ModaMes")
  return df.join(modaRenamed,Seq("Month"))
}

//dfLimpioMes.filter(col("ModaMes")==="NA").show()

def cambiarValorNAModa(atributo:String, df: DataFrame): DataFrame = {
  // Creamos columna temporal para las modas
  val dfModa = df.withColumn("ModaLocMes", modaLocMes(atributo,df))
  // Cuando sea NA, ponemos la moda
  val dfModa2 = modaMes(atributo,dfModa)
  val dfLimpioMes = dfModa2.withColumn(atributo, 
    when(col(atributo)==="NA", 
      col("ModaLocMes"))
      .otherwise(col(atributo)))
    .drop("ModaLocMes")
  // En caso de que siga habiaendo NAs, es la moda total del mes
  val dfLimpioTotal = dfLimpioMes.withColumn(atributo,
    when(col(atributo)==="NA",
      col("ModaMes"))
      .otherwise(col(atributo)))
      .drop("ModaMes")
  println(dfLimpioTotal.filter(col(atributo)==="NA").count())
  return dfLimpioTotal
}

val trainCleanCloud1 = cambiarValorNAModa("Cloud9am", trainM3)
println(trainCleanCloud1.filter(col("Cloud9am")==="NA").count())
val trainCleanCloud2 = cambiarValorNAModa("Cloud3pm", trainCleanCloud1)
println(trainCleanCloud2.filter(col("Cloud9am")==="NA").count()) 


val testCleanCloud1 = cambiarValorNAModa("Cloud9am", testM3)
val testCleanCloud2 = cambiarValorNAModa("Cloud3pm", testCleanCloud1)

val trainM4 = trainCleanCloud2
val testM4 = testCleanCloud2






/// EVAPORATION

def medianaLocMes(atributo: String, df: DataFrame): DataFrame = {

  val mediana = df.filter(col(atributo)=!= "NA").groupBy("Location","Month").agg(expr("percentile_approx("+atributo+",0.5)").alias("mediana"))
  
  val dfConMediana = df.join(mediana, Seq("Location", "Month"), "left")
  return dfConMediana
}

def medianaMes(atributo: String, df: DataFrame): DataFrame = {

  val mediana = df.filter(col(atributo)=!= "NA").groupBy("Month").agg(expr("percentile_approx("+atributo+",0.5)").alias("medianaTotal"))
  
  val dfConMediana = df.join(mediana, Seq("Month"), "left")
  return dfConMediana
}


def cambiarValorNAMediana(atributo:String, df: DataFrame): DataFrame = {
  // Creamos columna temporal para las medianas
  val dfMediana = medianaLocMes(atributo,df)
  val dfMedianaTotal = medianaMes(atributo,dfMediana)
  // Cuando sea NA, ponemos la mediana
  val dfLimpioMes = dfMedianaTotal.withColumn(atributo, 
    when(col(atributo)==="NA", 
      col("mediana"))
      .otherwise(col(atributo)))
    .drop("mediana")
  // En caso de que siga habiaendo NAs, es la mediana total
  val dfLimpioTotal = dfLimpioMes.withColumn(atributo,
    when(col(atributo)==="NA",
      col("medianaTotal"))
      .otherwise(col(atributo)))
      .drop("medianaTotal")
  println("NAs resultantes: "+dfLimpioTotal.filter(col(atributo)==="NA").count())
  dfLimpioTotal
}

val trainImp_base = cambiarValorNAMediana("Evaporation", trainM4)
val testImp_base= cambiarValorNAMediana("Evaporation", testM4)

// -------- 9) DESCARTE por vectores (0,0) en alguna WindDir* --------
def anyZeroVector(df: DataFrame): Column =
  windCatCols.map(c => (col(s"${c}_ux") === 0.0 && col(s"${c}_uy") === 0.0)).reduceOption(_ || _).getOrElse(lit(false))

val badTrainWind = trainImp_base.filter(anyZeroVector(trainImp_base)).count()
val badTestWind  = testImp_base.filter(anyZeroVector(testImp_base)).count()

val trainImp = trainImp_base.filter(!anyZeroVector(trainImp_base))
val testImp  = testImp_base.filter(!anyZeroVector(testImp_base))

// ========================================
// LIMPIEZA FINAL + MÉTRICAS
// ========================================

// 1) Cloud == 9
 
val trainClean = trainImp.filter(!(col("Cloud9am").cast("int") === 9 || col("Cloud3pm").cast("int") === 9))
val testClean  = testImp.filter(!(col("Cloud9am").cast("int") === 9 || col("Cloud3pm").cast("int") === 9))


val trainTotal = trainClean.count()
// 2) Evaporation > 40
val trainFinal = trainClean.orderBy(desc("Evaporation")).limit(trainTotal.toInt - (trainTotal/1000).toInt)

val testTotal = testClean.count()
val testFinal = testClean.orderBy(asc("Evaporation")).limit(testTotal.toInt-(testTotal/1000).toInt.toDouble)


// 3) Métricas (TEST)
val NTAIE_base   = testImp_base.count() - testImp.count()     // (0,0)
val NTAIE_cloud  = testImp.count()     - testClean.count()    // Cloud=9
val NTOE_evap    = testClean.count()   - testFinal.count()    // Evap>40
val NTAIE = NTAIE_base + NTAIE_cloud
val NTOE  = NTOE_evap

val totalTest = testDF.count()
val tasaNoClasificados = (NTAIE + NTOE).toDouble / totalTest

println("\n=== RESUMEN DE LIMPIEZA EN TEST ===")
println(s"NTAIE (ausentes/inconsistentes eliminados, incl. viento (0,0)): $NTAIE")
println(s"  - De ellos, por vectores de viento (0,0): $badTestWind")
println(s"NTOE (outliers eliminados): $NTOE")
println(f"Tasa de no clasificados: ${tasaNoClasificados * 100}%.4f%%")





// ======================================================================
// SELECCIÓN Y TRANSFORMACIÓN DE ATRIBUTOS
// - Eliminar atributos no informativos / redundantes
// - Construcción de nuevas variables agregadas
// - Codificación cíclica de Month (evitar ordinalidad)
// - Mantener viento como parámetros (ux, uy) ya generados
// - Discretización de continuas (≤ 15 bins) ajustada SOLO en TRAIN
// - Ensamblar "features" + "label"
// ======================================================================

import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector

// -----------------------------
// 0) Etiqueta (label) y copias base
// -----------------------------
val labelIndexer = new StringIndexer()
  .setInputCol("RainTomorrow")
  .setOutputCol("label")
  .setHandleInvalid("skip") // fit SOLO en train, luego transform test

// Empezamos desde los datos limpios finales
val trainBase = trainFinal
val testBase  = testFinal

// ----------------------------------------------------
// 1) Construcción de atributos y eliminación de ruido
// ----------------------------------------------------
// 1.1 Agregados medios para pares redundantes (tu metodología):
//     - Temperatura: promedio día (Temp9am/Temp3pm)
//     - Presión: promedio día (Pressure9am/Pressure3pm)
//     - Humedad: promedio día (Humidity9am/Humidity3pm)
//     - Viento: promedio día (WindSpeed9am/WindSpeed3pm)
//     - Nube: promedio día (Cloud9am/Cloud3pm) -> discreto (entero)
def buildAggregates(df: DataFrame): DataFrame = {
  val withTemps = if (df.columns.contains("Temp9am") && df.columns.contains("Temp3pm"))
    df.withColumn("Temp_avg", (col("Temp9am") + col("Temp3pm")) / 2.0)
  else df

  val withPress = if (df.columns.contains("Pressure9am") && df.columns.contains("Pressure3pm"))
    withTemps.withColumn("Pressure_avg", (col("Pressure9am") + col("Pressure3pm")) / 2.0)
  else withTemps

  val withHum = if (df.columns.contains("Humidity9am") && df.columns.contains("Humidity3pm"))
    withPress.withColumn("Humidity_avg", (col("Humidity9am") + col("Humidity3pm")) / 2.0)
  else withPress

  val withWindS = if (df.columns.contains("WindSpeed9am") && df.columns.contains("WindSpeed3pm"))
    withHum.withColumn("WindSpeed_avg", (col("WindSpeed9am") + col("WindSpeed3pm")) / 2.0)
  else withHum

  val withCloud = if (df.columns.contains("Cloud9am") && df.columns.contains("Cloud3pm"))
    withWindS.withColumn("Cloud_avg", floor( (col("Cloud9am") + col("Cloud3pm")) / 2.0 ))
  else withWindS

  withCloud
}

val trainAgg = buildAggregates(trainBase)
val testAgg  = buildAggregates(testBase)

def withCyclicMonth(df: DataFrame): DataFrame = {
  if (df.columns.contains("Month")) {
    val twoPi = lit(2.0 * math.Pi)
    df.withColumn("Month_sin", sin( twoPi * (col("Month").cast("double") / lit(12.0)) ))
      .withColumn("Month_cos", cos( twoPi * (col("Month").cast("double") / lit(12.0)) ))
  } else df
}

// vuelve a ejecutar estas dos líneas después de definir la función
val trainCyc = withCyclicMonth(trainAgg)
val testCyc  = withCyclicMonth(testAgg)


// 1.3 Columnas de viento: ya están como parámetros (ux,uy) creados antes:
//     WindGustDir_ux/_uy, WindDir9am_ux/_uy, WindDir3pm_ux/_uy (según tu bloque de viento).
val windVecCols =
  Seq("WindGustDir_ux","WindGustDir_uy","WindDir9am_ux","WindDir9am_uy","WindDir3pm_ux","WindDir3pm_uy")
    .filter(trainCyc.columns.contains)

// 1.4 Eliminar atributos no informativos o de alto riesgo de sobreajuste (según tu memoria):
//     - RainToday (redundante con Rainfall) 
//     - Location y Date (solo para imputaciones; ruido si se codifican naive)
//     - Sunshine (redundante vs Cloud; más ausentes)
//     - Columnas originales redundantes cuyo promedio ya usamos
val dropCols = Seq(
  "RainToday", "Location", "Date", "Sunshine",
  "Temp9am","Temp3pm",
  "Pressure9am","Pressure3pm",
  "Humidity9am","Humidity3pm",
  "WindSpeed9am","WindSpeed3pm",
  "Cloud9am","Cloud3pm",
  "Month" // ya usamos Month_sin/Month_cos
).filter(trainCyc.columns.contains)

val trainSel0 = trainCyc.drop(dropCols:_*)
val testSel0  = testCyc.drop(dropCols:_*)
// ----------------------------------------------------
// 2) Discretización de atributos continuos (≤ 15 bins)
//    (ajuste solo con TRAIN; aplicar en TEST)
//    * Ajustes: ensambladores tolerantes y discretización más ligera
// ----------------------------------------------------
val continuousCols = Seq(
  "MinTemp","MaxTemp","Temp_avg",
  "Rainfall","Evaporation",
  "WindGustSpeed","WindSpeed_avg",
  "Humidity_avg",
  "Pressure_avg"
  // "Cloud_avg" // (opcional) Si quieres discretizarla, descomenta esta línea.
  // Por si quedara alguna original (no debería tras drops, pero no molesta):
  // "Temp9am","Temp3pm","Pressure9am","Pressure3pm","Humidity9am","Humidity3pm","WindSpeed9am","WindSpeed3pm"
).distinct.filter(trainSel0.columns.contains)

// --- Cast robusto a Double de todas las continuas en TRAIN/TEST ---
import org.apache.spark.sql.types._
def castContinuousToDouble(df: DataFrame, cols: Seq[String]): DataFrame =
  cols.foldLeft(df){ (acc,c) => acc.withColumn(c, col(c).cast(DoubleType)) }

val trainSel = castContinuousToDouble(trainSel0, continuousCols).cache()
trainSel.count()  // materializa cache antes del fit
val testSel  = castContinuousToDouble(testSel0,  continuousCols)

// Discretizamos por cuantiles (más liviano para memoria)

val distinctByCol = continuousCols.map(c => c -> trainSel.select(c).distinct().count()).toMap
def bucketsFor(c: String) = math.min(10, math.max(3, distinctByCol(c).toInt - 1))
val discretizers = continuousCols.map { c =>
  new QuantileDiscretizer()
    .setInputCol(c)
    .setOutputCol(s"${c}_bin")
    .setNumBuckets(bucketsFor(c))
    .setRelativeError(0.02)
    .setHandleInvalid("keep")
}
 
// ----------------------------------------------------
// 3) Estandarización opcional de continuas (para modelos lineales)
// ----------------------------------------------------
val scalerInput = (continuousCols ++ windVecCols ++ Seq("Month_sin","Month_cos"))
  .distinct.filter(trainSel.columns.contains)

val assemblerCont = new VectorAssembler()
  .setInputCols(scalerInput.toArray)
  .setOutputCol("features_cont_raw")
  .setHandleInvalid("keep")   // <-- tolerante a null/NaN

val scaler = new StandardScaler()
  .setInputCol("features_cont_raw")
  .setOutputCol("features_cont_scaled")
  .setWithMean(true)
  .setWithStd(true)

// ----------------------------------------------------
// 4) Ensamblado de versión DISCRETIZADA (para NB/árboles si quieres)
// ----------------------------------------------------
val binCols = continuousCols.map(c => s"${c}_bin")
val assemblerBins = new VectorAssembler()
  .setInputCols((binCols ++ windVecCols ++ Seq("Month_sin","Month_cos"))
                 .filter(trainSel.columns.contains).toArray)
  .setOutputCol("features_bin")
  .setHandleInvalid("keep")   // <-- tolerante a null/NaN

// ----------------------------------------------------
// 5) Pipeline de transformación (fit SOLO en TRAIN)
// ----------------------------------------------------
val stages = (discretizers :+ assemblerCont :+ scaler :+ assemblerBins :+ labelIndexer).toArray
val ftPipeline = new Pipeline().setStages(stages)

// Ajuste en TRAIN (usar trainSel ya casteado y cacheado)
val ftModel: PipelineModel = ftPipeline.fit(trainSel)

// Transformar TRAIN/TEST
val trainFT = ftModel.transform(trainSel)
val testFT  = ftModel.transform(testSel)

// Salidas habituales
val train_for_linear = trainFT.select("label","features_cont_scaled").withColumnRenamed("features_cont_scaled","features")
val test_for_linear  = testFT.select("label","features_cont_scaled").withColumnRenamed("features_cont_scaled","features")

val train_for_nb_or_trees = trainFT.select("label","features_bin").withColumnRenamed("features_bin","features")
val test_for_nb_or_trees  = testFT.select("label","features_bin").withColumnRenamed("features_bin","features")
