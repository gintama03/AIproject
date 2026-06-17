import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const KyAiRagChatbotApp());
}

class KyAiRagChatbotApp extends StatelessWidget {
  const KyAiRagChatbotApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'KY AI RAG Chatbot',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF006D55),
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xFFF7F8FA),
      ),
      home: const ChatHomePage(),
    );
  }
}

class ChatHomePage extends StatefulWidget {
  const ChatHomePage({super.key});

  @override
  State<ChatHomePage> createState() => _ChatHomePageState();
}

class _ChatHomePageState extends State<ChatHomePage> {
  static const _ragApiBaseUrl = String.fromEnvironment(
    'RAG_API_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );

  static const _suggestedQuestions = [
    '2025 인공지능학과 졸업 최소 학점은?',
    '2022 의료인공지능학과 단일전공 전공 학점은?',
    '인공지능학과 졸업 후 진로는?',
  ];

  final _controller = TextEditingController();
  final _scrollController = ScrollController();
  final _apiClient = const RagApiClient(baseUrl: _ragApiBaseUrl);
  final _expandedSourceMessageIndexes = <int>{};
  final _messages = <ChatMessage>[
    const ChatMessage(
      text: '안녕하세요. 건양대학교 인공지능학과 RAG 챗봇입니다. 교육과정, 졸업요건, 진로, 입학 정보에 대해 질문해 주세요.',
      isUser: false,
      sources: [],
    ),
  ];

  bool _isCheckingApi = true;
  bool _isAnswering = false;
  bool _apiReady = false;
  int _documentCount = 0;
  String? _apiError;

  @override
  void initState() {
    super.initState();
    _checkApi();
  }

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _checkApi() async {
    setState(() {
      _isCheckingApi = true;
      _apiError = null;
    });

    try {
      final health = await _apiClient.health();
      setState(() {
        _apiReady = health.ok;
        _documentCount = health.documents;
        _apiError = health.ok ? null : 'RAG API 상태를 확인할 수 없습니다.';
      });
    } catch (error) {
      setState(() {
        _apiReady = false;
        _documentCount = 0;
        _apiError = 'RAG API 서버에 연결되지 않았습니다.';
      });
    } finally {
      setState(() {
        _isCheckingApi = false;
      });
    }
  }

  Future<void> _ask([String? preset]) async {
    final question = (preset ?? _controller.text).trim();
    if (question.isEmpty || _isAnswering) {
      return;
    }

    setState(() {
      _messages.add(ChatMessage(text: question, isUser: true, sources: const []));
      _controller.clear();
      _isAnswering = true;
    });
    _scrollToBottom();

    try {
      final response = await _apiClient.ask(question);
      setState(() {
        _apiReady = true;
        _apiError = null;
        _messages.add(
          ChatMessage(
            text: response.answer,
            isUser: false,
            sources: response.sources,
            mode: response.mode,
            apiError: response.error,
          ),
        );
      });
    } catch (error) {
      setState(() {
        _apiReady = false;
        _apiError = 'RAG API 서버에 연결되지 않았습니다.';
        _messages.add(
          ChatMessage(
            text:
                'RAG API 서버에 연결하지 못했습니다.\n\nPC에서 `scripts\\run_rag_api_server.ps1`를 실행한 뒤 다시 질문해 주세요. 휴대폰에서 실행 중이면 `adb reverse tcp:8000 tcp:8000` 설정도 필요할 수 있습니다.',
            isUser: false,
            sources: const [],
            mode: 'error',
            apiError: error.toString(),
          ),
        );
      });
    } finally {
      setState(() {
        _isAnswering = false;
      });
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) {
        return;
      }
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 240),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.white,
        surfaceTintColor: Colors.white,
        titleSpacing: 16,
        title: const _AppTitle(),
      ),
      body: SafeArea(
        child: Column(
          children: [
            _StatusPanel(
              documentCount: _documentCount,
              isCheckingApi: _isCheckingApi,
              isAnswering: _isAnswering,
              apiReady: _apiReady,
              error: _apiError,
              apiBaseUrl: _ragApiBaseUrl,
              onReload: _checkApi,
            ),
            Expanded(
              child: ListView(
                controller: _scrollController,
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
                children: [
                  const _WelcomeBlock(),
                  const SizedBox(height: 14),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      for (final question in _suggestedQuestions)
                        _QuestionChip(label: question, onPressed: _ask),
                    ],
                  ),
                  const SizedBox(height: 18),
                  for (var index = 0; index < _messages.length; index++)
                    ..._buildMessageWidgets(index),
                ],
              ),
            ),
            _InputBar(
              controller: _controller,
              enabled: !_isAnswering,
              onSubmitted: _ask,
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildMessageWidgets(int index) {
    final message = _messages[index];
    final hasSources = !message.isUser && message.sources.isNotEmpty;
    final isSourceExpanded = _expandedSourceMessageIndexes.contains(index);

    return [
      _ChatBubble(message: message),
      if (hasSources) ...[
        const SizedBox(height: 6),
        _SourceToggleButton(
          isExpanded: isSourceExpanded,
          sourceCount: message.sources.length > 3 ? 3 : message.sources.length,
          mode: message.mode,
          onPressed: () {
            setState(() {
              if (isSourceExpanded) {
                _expandedSourceMessageIndexes.remove(index);
              } else {
                _expandedSourceMessageIndexes.add(index);
              }
            });
          },
        ),
      ],
      const SizedBox(height: 10),
      if (hasSources && isSourceExpanded) ...[
        _SourceList(results: message.sources),
        const SizedBox(height: 10),
      ],
    ];
  }
}

class ApiHealth {
  const ApiHealth({
    required this.ok,
    required this.documents,
    required this.model,
    required this.llmReady,
  });

  factory ApiHealth.fromJson(Map<String, dynamic> json) {
    return ApiHealth(
      ok: json['ok'] == true,
      documents: int.tryParse(json['documents']?.toString() ?? '') ?? 0,
      model: json['model']?.toString() ?? '',
      llmReady: json['llm_ready'] == true,
    );
  }

  final bool ok;
  final int documents;
  final String model;
  final bool llmReady;
}

class RagDocument {
  const RagDocument({
    required this.docId,
    required this.title,
    required this.content,
    required this.category,
    required this.section,
    required this.sourceFile,
    required this.sourceGroup,
    required this.year,
    required this.url,
  });

  factory RagDocument.fromJson(Map<String, dynamic> json) {
    return RagDocument(
      docId: json['doc_id']?.toString() ?? '',
      title: json['title']?.toString() ?? '',
      content: json['content']?.toString() ?? '',
      category: json['category']?.toString() ?? '',
      section: json['section']?.toString() ?? '',
      sourceFile: json['source_file']?.toString() ?? '',
      sourceGroup: json['source_group']?.toString() ?? '',
      year: json['year']?.toString() ?? '',
      url: json['url']?.toString() ?? '',
    );
  }

  final String docId;
  final String title;
  final String content;
  final String category;
  final String section;
  final String sourceFile;
  final String sourceGroup;
  final String year;
  final String url;
}

class SearchResult {
  const SearchResult({required this.document, required this.score});

  factory SearchResult.fromJson(Map<String, dynamic> json) {
    return SearchResult(
      document: RagDocument.fromJson(json),
      score: double.tryParse(json['score']?.toString() ?? '') ?? 0,
    );
  }

  final RagDocument document;
  final double score;
}

class ChatMessage {
  const ChatMessage({
    required this.text,
    required this.isUser,
    required this.sources,
    this.mode = '',
    this.apiError = '',
  });

  final String text;
  final bool isUser;
  final List<SearchResult> sources;
  final String mode;
  final String apiError;
}

class RagApiResponse {
  const RagApiResponse({
    required this.answer,
    required this.sources,
    required this.mode,
    required this.error,
  });

  factory RagApiResponse.fromJson(Map<String, dynamic> json) {
    final rawSources = json['sources'];
    final sources = rawSources is List
        ? rawSources
            .whereType<Map>()
            .map((source) => SearchResult.fromJson(Map<String, dynamic>.from(source)))
            .toList(growable: false)
        : <SearchResult>[];

    return RagApiResponse(
      answer: cleanAnswerText(json['answer']?.toString() ?? ''),
      sources: sources,
      mode: json['mode']?.toString() ?? '',
      error: json['error']?.toString() ?? '',
    );
  }

  final String answer;
  final List<SearchResult> sources;
  final String mode;
  final String error;
}

String cleanAnswerText(String value) {
  final lines = value
      .split('\n')
      .where((line) => !RegExp(r'^\s*(근거|출처)\s*:').hasMatch(line))
      .toList(growable: false);
  return lines.join('\n').trim();
}

class RagApiClient {
  const RagApiClient({required this.baseUrl});

  final String baseUrl;

  Uri _uri(String path) {
    return Uri.parse('${baseUrl.replaceAll(RegExp(r"/+$"), '')}$path');
  }

  Future<ApiHealth> health() async {
    final response = await http.get(_uri('/health')).timeout(const Duration(seconds: 6));
    if (response.statusCode != 200) {
      throw Exception('HTTP ${response.statusCode}: ${response.body}');
    }
    final decoded = jsonDecode(utf8.decode(response.bodyBytes));
    if (decoded is! Map<String, dynamic>) {
      throw Exception('API health 응답 형식이 올바르지 않습니다.');
    }
    return ApiHealth.fromJson(decoded);
  }

  Future<RagApiResponse> ask(String question) async {
    final response = await http
        .post(
          _uri('/ask'),
          headers: const {'Content-Type': 'application/json; charset=utf-8'},
          body: jsonEncode({
            'question': question,
            'top_k': 3,
            'use_llm': true,
          }),
        )
        .timeout(const Duration(seconds: 75));

    if (response.statusCode != 200) {
      throw Exception('HTTP ${response.statusCode}: ${response.body}');
    }

    final decoded = jsonDecode(utf8.decode(response.bodyBytes));
    if (decoded is! Map<String, dynamic>) {
      throw Exception('API 답변 형식이 올바르지 않습니다.');
    }
    return RagApiResponse.fromJson(decoded);
  }
}

class _AppTitle extends StatelessWidget {
  const _AppTitle();

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: const Color(0xFFE4E7EC)),
          ),
          child: Padding(
            padding: const EdgeInsets.all(6),
            child: Image.asset(
              'assets/image/app_logo.png',
              fit: BoxFit.contain,
            ),
          ),
        ),
        const SizedBox(width: 10),
        const Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'KY AI Guide',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
              ),
              SizedBox(height: 2),
              Text(
                '건양대학교 인공지능학과 안내 챗봇',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(fontSize: 12, color: Color(0xFF667085)),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _StatusPanel extends StatelessWidget {
  const _StatusPanel({
    required this.documentCount,
    required this.isCheckingApi,
    required this.isAnswering,
    required this.apiReady,
    required this.error,
    required this.apiBaseUrl,
    required this.onReload,
  });

  final int documentCount;
  final bool isCheckingApi;
  final bool isAnswering;
  final bool apiReady;
  final String? error;
  final String apiBaseUrl;
  final VoidCallback onReload;

  @override
  Widget build(BuildContext context) {
    final statusLabel = isCheckingApi
        ? 'API 확인 중'
        : isAnswering
            ? '답변 생성 중'
            : apiReady
                ? 'RAG API 연결됨'
                : 'API 연결 필요';

    return Container(
      width: double.infinity,
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    _StatusPill(
                      icon: Icons.storage_outlined,
                      label: documentCount > 0 ? '문서 $documentCount개' : '문서 확인 대기',
                    ),
                    _StatusPill(icon: Icons.verified_outlined, label: statusLabel),
                    _StatusPill(icon: Icons.cloud_outlined, label: apiBaseUrl),
                  ],
                ),
              ),
              IconButton(
                tooltip: 'API 상태 다시 확인',
                onPressed: onReload,
                icon: const Icon(Icons.refresh),
              ),
            ],
          ),
          if (error != null) ...[
            const SizedBox(height: 6),
            Text(
              error!,
              style: const TextStyle(color: Color(0xFFB42318), fontSize: 12),
            ),
          ],
        ],
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFFEAF5F1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: const Color(0xFF006D55)),
          const SizedBox(width: 5),
          Text(
            label,
            style: const TextStyle(
              color: Color(0xFF006D55),
              fontWeight: FontWeight.w600,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}

class _WelcomeBlock extends StatelessWidget {
  const _WelcomeBlock();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFE4E7EC)),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '무엇을 도와드릴까요?',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
          ),
          SizedBox(height: 8),
          Text(
            '교육과정, 졸업요건, 학과소개, 진로 및 자격증, 입학 정보를 저장된 문서 기반으로 답변합니다.',
            style: TextStyle(color: Color(0xFF475467), height: 1.45),
          ),
        ],
      ),
    );
  }
}

class _QuestionChip extends StatelessWidget {
  const _QuestionChip({required this.label, required this.onPressed});

  final String label;
  final void Function(String question) onPressed;

  @override
  Widget build(BuildContext context) {
    return ActionChip(
      onPressed: () => onPressed(label),
      avatar: const Icon(Icons.add_comment_outlined, size: 17),
      label: Text(label),
      labelStyle: const TextStyle(fontSize: 13),
      backgroundColor: Colors.white,
      surfaceTintColor: Colors.white,
      side: const BorderSide(color: Color(0xFFD0D5DD)),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
    );
  }
}

class _ChatBubble extends StatelessWidget {
  const _ChatBubble({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final alignment = message.isUser ? Alignment.centerRight : Alignment.centerLeft;
    final color = message.isUser ? const Color(0xFF006D55) : Colors.white;
    final textColor = message.isUser ? Colors.white : const Color(0xFF101828);

    return Align(
      alignment: alignment,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 720),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(8),
            border: message.isUser ? null : Border.all(color: const Color(0xFFE4E7EC)),
          ),
          child: Text(
            message.text,
            style: TextStyle(color: textColor, fontSize: 14, height: 1.5),
          ),
        ),
      ),
    );
  }
}

class _SourceToggleButton extends StatelessWidget {
  const _SourceToggleButton({
    required this.isExpanded,
    required this.sourceCount,
    required this.mode,
    required this.onPressed,
  });

  final bool isExpanded;
  final int sourceCount;
  final String mode;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    final modeLabel = mode == 'gemini'
        ? 'AI 답변'
        : mode == 'local_fallback'
            ? '로컬 RAG'
            : 'RAG';

    return Align(
      alignment: Alignment.centerLeft,
      child: TextButton.icon(
        onPressed: onPressed,
        icon: Icon(isExpanded ? Icons.menu_book : Icons.menu_book_outlined),
        label: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              isExpanded ? '출처 숨기기' : '출처',
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
            const SizedBox(width: 8),
            Text(
              '$sourceCount개 · $modeLabel',
              style: const TextStyle(
                color: Color(0xFF667085),
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(width: 4),
            Icon(
              isExpanded ? Icons.expand_less : Icons.expand_more,
              size: 18,
              color: const Color(0xFF667085),
            ),
          ],
        ),
        style: TextButton.styleFrom(
          foregroundColor: const Color(0xFF344054),
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          minimumSize: Size.zero,
          tapTargetSize: MaterialTapTargetSize.shrinkWrap,
        ),
      ),
    );
  }
}

class _SourceList extends StatelessWidget {
  const _SourceList({required this.results});

  final List<SearchResult> results;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        for (var index = 0; index < results.length.clamp(0, 3); index++)
          Padding(
            padding: EdgeInsets.only(bottom: index == 2 ? 0 : 8),
            child: _SourceCard(rank: index + 1, result: results[index]),
          ),
      ],
    );
  }
}

class _SourceCard extends StatelessWidget {
  const _SourceCard({required this.rank, required this.result});

  final int rank;
  final SearchResult result;

  @override
  Widget build(BuildContext context) {
    final doc = result.document;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFFCF5),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFFEC84B)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(
                Icons.article_outlined,
                size: 19,
                color: Color(0xFFB54708),
              ),
              const SizedBox(width: 7),
              Expanded(
                child: Text(
                  '근거 $rank · ${doc.category}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontWeight: FontWeight.w800,
                    color: Color(0xFFB54708),
                  ),
                ),
              ),
              Text(
                'score ${result.score.toStringAsFixed(2)}',
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFFB54708),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            doc.title,
            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 4),
          Text(
            '${doc.sourceFile}${doc.year.isEmpty ? '' : ' · ${doc.year}'}',
            style: const TextStyle(fontSize: 12, color: Color(0xFF475467)),
          ),
        ],
      ),
    );
  }
}

class _InputBar extends StatelessWidget {
  const _InputBar({
    required this.controller,
    required this.enabled,
    required this.onSubmitted,
  });

  final TextEditingController controller;
  final bool enabled;
  final VoidCallback onSubmitted;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: Color(0xFFE4E7EC))),
      ),
      child: Row(
        children: [
          const Icon(Icons.search, color: Color(0xFF667085)),
          const SizedBox(width: 8),
          Expanded(
            child: Shortcuts(
              shortcuts: const <ShortcutActivator, Intent>{
                SingleActivator(LogicalKeyboardKey.enter): _SubmitQuestionIntent(),
                SingleActivator(LogicalKeyboardKey.numpadEnter): _SubmitQuestionIntent(),
              },
              child: Actions(
                actions: <Type, Action<Intent>>{
                  _SubmitQuestionIntent: CallbackAction<_SubmitQuestionIntent>(
                    onInvoke: (_) {
                      if (enabled) {
                        onSubmitted();
                      }
                      return null;
                    },
                  ),
                },
                child: TextField(
                  controller: controller,
                  enabled: enabled,
                  minLines: 1,
                  maxLines: 4,
                  keyboardType: TextInputType.multiline,
                  textInputAction: TextInputAction.send,
                  onSubmitted: (_) => onSubmitted(),
                  decoration: const InputDecoration(
                    hintText: '학과 정보에 대해 질문하기',
                    filled: true,
                    fillColor: Color(0xFFF2F4F7),
                    border: OutlineInputBorder(
                      borderSide: BorderSide.none,
                      borderRadius: BorderRadius.all(Radius.circular(8)),
                    ),
                    contentPadding: EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 12,
                    ),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),
          FilledButton(
            onPressed: enabled ? onSubmitted : null,
            style: FilledButton.styleFrom(
              minimumSize: const Size(46, 46),
              padding: EdgeInsets.zero,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
            ),
            child: const Icon(Icons.send_rounded),
          ),
        ],
      ),
    );
  }
}

class _SubmitQuestionIntent extends Intent {
  const _SubmitQuestionIntent();
}
