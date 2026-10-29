const fs = require('fs');
const path = require('path');
const db = require('../db.js');
const rag = require('../rag.js');

async function runEvaluation() {
  console.log('=== ЗАПУСК ОЦЕНКИ И КАЛИБРОВКИ RAG ===\n');

  // 1. Индексация документов из sample_docs/
  const sampleDocsDir = path.join(__dirname, '..', 'sample_docs');
  const files = fs.readdirSync(sampleDocsDir).filter(f => f.endsWith('.txt') || f.endsWith('.md'));

  console.log(`Индексация тестовых документов из ${sampleDocsDir}:`);
  for (const file of files) {
    const fullPath = path.join(sampleDocsDir, file);
    const text = fs.readFileSync(fullPath, 'utf-8');

    // Проверяем, есть ли уже такой документ
    const existing = db.listDocuments().find(d => d.title === file);
    if (existing) {
      console.log(`- Документ "${file}" уже в базе (ID: ${existing.id}), переиндексация...`);
      await rag.reindexDocument(existing.id);
    } else {
      console.log(`- Добавление нового документа "${file}"...`);
      const res = await rag.indexDocument(file, text);
      console.log(`  ✓ Добавлен (ID: ${res.id}, фрагментов: ${res.chunkCount})`);
    }
  }

  // 2. Чтение вопросов
  const questionsPath = path.join(__dirname, 'questions.json');
  const questions = JSON.parse(fs.readFileSync(questionsPath, 'utf-8'));
  console.log(`\nЗагружено тестовых вопросов: ${questions.length}\n`);

  let allPassed = true;
  const results = [];

  for (const q of questions) {
    console.log(`--------------------------------------------------------------------------------`);
    console.log(`Вопрос #${q.id} [${q.trap ? 'ЛОВУШКА' : 'ОБЫЧНЫЙ ВОПРОС'}]:`);
    console.log(`"${q.question}"`);

    const res = await rag.askQuestion(q.question);
    const sim = res.similarity.toFixed(4);
    const thresh = res.threshold.toFixed(4);

    console.log(`> Сходство (Similarity): ${sim} | Порог (Threshold): ${thresh}`);
    console.log(`> Вызов модели LLM: ${res.modelCalled ? 'ДА' : 'НЕТ (отсечено порогом)'}`);
    console.log(`> Источники: ${res.sources && res.sources.length ? res.sources.join(', ') : 'нет'}`);
    console.log(`> Ответ бота:\n${res.answer}\n`);

    if (q.trap) {
      if (res.modelCalled || res.similarity >= res.threshold) {
        console.log(`❌ ОШИБКА: Ловушка НЕ была отсечена порогом сходства!`);
        allPassed = false;
      } else {
        console.log(`✓ УСПЕХ: Ловушка корректно отсечена (сходство ${sim} < порога ${thresh}), модель не вызывалась.`);
      }
    } else {
      if (!res.modelCalled || res.similarity < res.threshold) {
        console.log(`❌ ОШИБКА: Обычный вопрос был ложно отсечен порогом сходства!`);
        allPassed = false;
      } else {
        console.log(`✓ УСПЕХ: Вопрос корректно распознан (сходство ${sim} >= порога ${thresh}), источники найдены.`);
      }
    }

    results.push({
      id: q.id,
      trap: q.trap,
      similarity: res.similarity,
      threshold: res.threshold,
      modelCalled: res.modelCalled,
      sources: res.sources
    });
  }

  console.log(`\n================== ИТОГИ КАЛИБРОВКИ ==================`);
  console.table(results.map(r => ({
    'ID': r.id,
    'Тип': r.trap ? 'Ловушка' : 'Обычный',
    'Сходство': r.similarity.toFixed(4),
    'Порог': r.threshold.toFixed(4),
    'Модель вызвана': r.modelCalled ? 'Да' : 'Нет',
    'Источники': r.sources.join(', ') || 'нет'
  })));

  if (allPassed) {
    console.log('\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!');
    process.exit(0);
  } else {
    console.error('\n⚠️ ЕСТЬ ОШИБКИ КАЛИБРОВКИ!');
    process.exit(1);
  }
}

runEvaluation().catch(err => {
  console.error('Критическая ошибка оценки:', err);
  process.exit(1);
});

