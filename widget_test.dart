import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:ky_ai_rag_chatbot/main.dart';

void main() {
  testWidgets('shows the chatbot home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const KyAiRagChatbotApp());
    await tester.pumpAndSettle();

    expect(find.text('KY AI Guide'), findsOneWidget);
    expect(find.text('무엇을 도와드릴까요?'), findsOneWidget);
    expect(find.textContaining('RAG API'), findsWidgets);
    expect(find.byIcon(Icons.send_rounded), findsOneWidget);
  });

  testWidgets('shows a connection message when the API is unavailable', (
    WidgetTester tester,
  ) async {
    const question = '2025 인공지능학과 졸업 최소 학점은?';

    await tester.pumpWidget(const KyAiRagChatbotApp());
    await tester.pumpAndSettle();

    await tester.enterText(find.byType(TextField), question);
    await tester.tap(find.byIcon(Icons.send_rounded));
    await tester.pumpAndSettle();

    expect(find.text(question), findsWidgets);
    expect(find.textContaining('RAG API 서버에 연결하지 못했습니다.'), findsOneWidget);
  });
}
