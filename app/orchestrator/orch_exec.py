import logging

import matplotlib
from pandas import DataFrame

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.agent.chart_gen_agent import ChartAgent
from app.agent.fall_back import GeneralAnswerAgent
from app.agent.schema_info_agent import SchemaInfoAgent
from app.agent.sql_generation_agent import SQLGenerationAgent
from app.agent.table_identification_agent import TableIdentiferAgent
from app.router.route_improvisation import Router
from app.utils.char_context import ChartContext
from app.utils.sql_executor import SqlExec
from app.utils.sql_validator import validator

logger = logging.getLogger(__name__)


######Simple Hardcoded Orchetrator########
class Orchestrator:
    def __init__(self, user_question: str):
        self.user_question = user_question
        self.TA = TableIdentiferAgent(self.user_question)
        self.router = Router()

    def executor(self):
        logger.info("Running orchestration for question: %s", self.user_question)
        decision_maker = self.router.route_question(self.user_question)
        logger.info("Router decision: %s", decision_maker)
        if decision_maker.route == "sql_query":
            ta_agent_result = self.TA.run()
            logger.info("Table identifier result: %s", ta_agent_result)
            sql_gen_agent = SQLGenerationAgent(
                self.user_question, ta_agent_result["relevant_tables"]
            )
            sql_gen_res = sql_gen_agent.run()
            logger.info("Generated SQL: %s", sql_gen_res)
            sql_val = validator(sql_gen_res)
            if sql_val is True:
                logger.info("Executing generated SQL")
                sql_exec = SqlExec().sql_executor(sql_gen_res)
                return sql_exec
            else:
                logger.warning("SQL validation failed")
        elif decision_maker.route == "sql_and_chart":
            ta_agent_result = self.TA.run()
            logger.info("Table identifier result: %s", ta_agent_result)
            sql_gen_agent = SQLGenerationAgent(
                self.user_question, ta_agent_result["relevant_tables"]
            )
            sql_gen_res = sql_gen_agent.run()
            logger.info("Generated SQL: %s", sql_gen_res)
            sql_val = validator(sql_gen_res)
            if sql_val is True:
                logger.info("Executing generated SQL for chart flow")
                sql_exec: DataFrame | str = SqlExec().sql_executor(sql_gen_res)
                if not isinstance(sql_exec, DataFrame):
                    logger.warning(
                        "SQL execution did not return a DataFrame: %s",
                        repr(sql_exec),
                    )
                    return {
                        "type": "error",
                        "message": "Query execution failed or returned invalid result.",
                    }

                chart_context = ChartContext.build_chart_context(sql_exec)
                chart_gen_decision = ChartAgent().run(self.user_question, chart_context)
                logger.info("Chart generation decision: %s", chart_gen_decision)

                if (
                    chart_gen_decision.x_axis not in sql_exec.columns
                    or chart_gen_decision.y_axis not in sql_exec.columns
                ):
                    logger.warning(
                        "Chart axis missing in SQL result columns: %s",
                        sql_exec.columns.tolist(),
                    )
                    return {
                        "type": "error",
                        "message": "Chart axis names not present in query result.",
                    }

                fig, ax = plt.subplots()
                if chart_gen_decision.chart_type == "line":
                    ax.plot(
                        sql_exec[chart_gen_decision.x_axis],
                        sql_exec[chart_gen_decision.y_axis],
                    )
                elif chart_gen_decision.chart_type == "bar":
                    ax.bar(
                        sql_exec[chart_gen_decision.x_axis],
                        sql_exec[chart_gen_decision.y_axis],
                    )
                elif chart_gen_decision.chart_type == "scatter":
                    ax.scatter(
                        sql_exec[chart_gen_decision.x_axis],
                        sql_exec[chart_gen_decision.y_axis],
                    )
                elif chart_gen_decision.chart_type == "histogram":
                    ax.hist(sql_exec[chart_gen_decision.x_axis], bins=10)
                elif chart_gen_decision.chart_type == "pie":
                    pie_values = sql_exec[chart_gen_decision.y_axis]
                    if (pie_values < 0).any():
                        logger.warning("Pie chart not possible with negative values, falling back to bar chart")
                        ax.bar(
                            sql_exec[chart_gen_decision.x_axis],
                            pie_values,
                        )
                    else:
                        ax.pie(
                            pie_values,
                            labels=sql_exec[chart_gen_decision.x_axis].astype(str).tolist(),
                            autopct="%1.1f%%",
                        )
                elif chart_gen_decision.chart_type == "box":
                    ax.boxplot(sql_exec[chart_gen_decision.y_axis])
                elif chart_gen_decision.chart_type == "area":
                    ax.fill_between(
                        sql_exec[chart_gen_decision.x_axis],
                        sql_exec[chart_gen_decision.y_axis],  # type: ignore
                    )
                ax.set_xlabel(chart_gen_decision.x_axis)
                ax.set_ylabel(chart_gen_decision.y_axis)
                return {"type": "chart", "figure": fig, "data": sql_exec}
            else:
                logger.warning("SQL validation failed for chart flow")

        elif decision_maker.route == "configuration":
            return (
                "Use the `/schema/config` FastAPI endpoints to manage column "
                "labels, descriptions, and assistant visibility."
            )
        elif decision_maker.route == "schema_info":
            schema_info_agent = SchemaInfoAgent(
                self.user_question, decision_maker.reasoning
            )
            res = schema_info_agent.run()
            logger.info("Schema info response generated")
            return res
        else:
            gen_ans_agent = GeneralAnswerAgent()
            res = gen_ans_agent.run("...", self.user_question, decision_maker.reasoning)
            return res


#### Unite Test 3 #########
# if __name__ == "__main__":
#     user_query = "visuzlie me year wise movie count"
#     print("..Orchestration Started.....")
#     orch_run = Orchestrator(user_query)
#     orch_run.executor()
#     print("..Orchetsration Completed.....")
